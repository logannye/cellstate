from __future__ import annotations

import hashlib
import importlib.util
import io
import json
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
_verify_loaded_image_payload = SCRIPT_MODULE._verify_loaded_image_payload
verify_archive = SCRIPT_MODULE.verify_archive
verify_archives = SCRIPT_MODULE.verify_archives
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
        "build": {
            "dockerfile_sha256": "0" * 64,
            "oci_output": "type=oci",
            "provenance_attestation_disabled": True,
            "reproducibility_build_count": 2,
            "requirements_sha256": "0" * 64,
            "source_date_epoch": 1_786_406_400,
        },
        "config_digest": config_digest,
        "image_digest": manifest_digest,
        "image_reference": f"synthetic-runtime@{manifest_digest}",
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


def test_two_independent_oci_archives_match_the_complete_locked_identity(tmp_path: Path) -> None:
    files, lock = _fixture_payloads()
    lock_path = tmp_path / "runtime-image-lock.json"
    first = tmp_path / "first.oci.tar"
    second = tmp_path / "second.oci.tar"
    _write_lock(lock_path, lock)
    _write_archive(first, files)
    _write_archive(second, files, reverse=True)

    identity = verify_archives(lock_path, [first, second])

    assert identity.index_digest == lock["oci_index_digest"]
    assert identity.image_digest == lock["image_digest"]
    assert identity.config_digest == lock["config_digest"]
    assert len(identity.layer_digests) == 1


def test_oci_verifier_rejects_blob_content_that_does_not_match_its_digest(tmp_path: Path) -> None:
    files, lock = _fixture_payloads()
    layer_path = next(name for name in files if files[name].startswith(b"source-free"))
    files[layer_path] = b"x" * len(files[layer_path])
    lock_path = tmp_path / "runtime-image-lock.json"
    archive_path = tmp_path / "tampered.oci.tar"
    _write_lock(lock_path, lock)
    _write_archive(archive_path, files)

    with pytest.raises(VerificationError, match="does not match path digest"):
        verify_archive(lock_path, archive_path)


def test_oci_verifier_rejects_unreferenced_blobs(tmp_path: Path) -> None:
    files, lock = _fixture_payloads()
    extra = b"unreferenced"
    files[_blob_path(_digest(extra))] = extra
    lock_path = tmp_path / "runtime-image-lock.json"
    archive_path = tmp_path / "extra.oci.tar"
    _write_lock(lock_path, lock)
    _write_archive(archive_path, files)

    with pytest.raises(VerificationError, match="missing or unreferenced blobs"):
        verify_archive(lock_path, archive_path)


def test_oci_verifier_rejects_nonregular_archive_members(tmp_path: Path) -> None:
    files, lock = _fixture_payloads()
    symlink = tarfile.TarInfo("blobs/sha256/" + "f" * 64)
    symlink.type = tarfile.SYMTYPE
    symlink.linkname = "../../index.json"
    lock_path = tmp_path / "runtime-image-lock.json"
    archive_path = tmp_path / "symlink.oci.tar"
    _write_lock(lock_path, lock)
    _write_archive(archive_path, files, extra_member=symlink)

    with pytest.raises(VerificationError, match="non-regular OCI archive member"):
        verify_archive(lock_path, archive_path)


def test_build_input_verifier_binds_hashes_platform_and_epoch(tmp_path: Path) -> None:
    files, lock = _fixture_payloads()
    del files
    context = tmp_path / "context"
    context.mkdir()
    dockerfile = context / "Dockerfile"
    requirements = context / "requirements.lock"
    dockerfile.write_bytes(b"FROM scratch\n")
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
