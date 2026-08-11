#!/usr/bin/env python3
"""Verify the frozen Item 12.2 OCI runtime without opening biological source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

OCI_INDEX_MEDIA_TYPE = "application/vnd.oci.image.index.v1+json"
OCI_MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"
OCI_CONFIG_MEDIA_TYPE = "application/vnd.oci.image.config.v1+json"
OCI_LAYOUT = {"imageLayoutVersion": "1.0.0"}
SHA256_DIGEST = re.compile(r"sha256:([0-9a-f]{64})\Z")


class VerificationError(RuntimeError):
    """Raised when a build input, OCI archive, or loaded image violates its lock."""


@dataclass(frozen=True)
class ArchiveIdentity:
    """Content identities that must agree across independent OCI builds."""

    index_digest: str
    image_digest: str
    config_digest: str
    layer_digests: tuple[str, ...]


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise VerificationError(f"{label} must be a JSON object")
    return parsed


def _load_lock(path: Path) -> dict[str, Any]:
    try:
        lock = _load_json_bytes(path.read_bytes(), str(path))
    except OSError as exc:
        raise VerificationError(f"cannot read runtime image lock {path}: {exc}") from exc
    if lock.get("runtime_image_lock_schema") != ("cellstate-sciplex3-v5-runtime-image-lock"):
        raise VerificationError("unexpected runtime image lock schema")
    if lock.get("runtime_image_lock_version") != "1.0.0":
        raise VerificationError("unexpected runtime image lock version")
    return lock


def _required_mapping(parent: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise VerificationError(f"{label}.{key} must be an object")
    return value


def _required_string(parent: Mapping[str, Any], key: str, label: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{label}.{key} must be a non-empty string")
    return value


def _required_digest(parent: Mapping[str, Any], key: str, label: str) -> str:
    value = _required_string(parent, key, label)
    if SHA256_DIGEST.fullmatch(value) is None:
        raise VerificationError(f"{label}.{key} must be a lowercase sha256 digest")
    return value


def _required_positive_int(parent: Mapping[str, Any], key: str, label: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise VerificationError(f"{label}.{key} must be a positive integer")
    return value


def verify_build_inputs(lock_path: Path, context: Path) -> int:
    """Verify the frozen Docker context and return its locked source epoch."""

    lock = _load_lock(lock_path)
    build = _required_mapping(lock, "build", "lock")
    expected_files = {
        "Dockerfile": _required_string(build, "dockerfile_sha256", "lock.build"),
        "requirements.lock": _required_string(build, "requirements_sha256", "lock.build"),
    }
    for name, expected in expected_files.items():
        if SHA256_DIGEST.fullmatch(f"sha256:{expected}") is None:
            raise VerificationError(f"locked {name} hash must be lowercase sha256 hex")
        path = context / name
        try:
            actual = _sha256_file(path)
        except OSError as exc:
            raise VerificationError(f"cannot hash frozen build input {path}: {exc}") from exc
        if actual != expected:
            raise VerificationError(
                f"frozen build input drift for {name}: expected {expected}, got {actual}"
            )

    if lock.get("platform") != "linux/amd64":
        raise VerificationError("runtime image lock platform must be linux/amd64")
    if lock.get("operating_system") != "linux" or lock.get("architecture") != "amd64":
        raise VerificationError("runtime image lock OS/architecture is inconsistent")
    if build.get("oci_output") != "type=oci":
        raise VerificationError("runtime image lock must require OCI output")
    if build.get("provenance_attestation_disabled") is not True:
        raise VerificationError("runtime image lock must disable provenance attestations")
    if build.get("reproducibility_build_count") != 2:
        raise VerificationError("runtime image lock must require exactly two independent builds")
    return _required_positive_int(build, "source_date_epoch", "lock.build")


class _OciArchive:
    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self._archive = tarfile.open(path, mode="r:*")  # noqa: SIM115
        except (OSError, tarfile.TarError) as exc:
            raise VerificationError(f"cannot open OCI archive {path}: {exc}") from exc
        self._members: dict[str, tarfile.TarInfo] = {}
        try:
            self._index_members()
        except Exception:
            self._archive.close()
            raise

    def close(self) -> None:
        self._archive.close()

    def __enter__(self) -> _OciArchive:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _index_members(self) -> None:
        for member in self._archive.getmembers():
            name = member.name.removeprefix("./")
            path = PurePosixPath(name)
            if not name or path.is_absolute() or ".." in path.parts:
                raise VerificationError(f"unsafe OCI archive member {member.name!r}")
            if member.isdir():
                continue
            if not member.isfile():
                raise VerificationError(f"non-regular OCI archive member {member.name!r}")
            if name in self._members:
                raise VerificationError(f"duplicate OCI archive member {name!r}")
            self._members[name] = member

        allowed_roots = {"index.json", "oci-layout"}
        for name in self._members:
            if name in allowed_roots:
                continue
            parts = PurePosixPath(name).parts
            if len(parts) != 3 or parts[:2] != ("blobs", "sha256"):
                raise VerificationError(f"unexpected OCI archive member {name!r}")
            if re.fullmatch(r"[0-9a-f]{64}", parts[2]) is None:
                raise VerificationError(f"invalid OCI blob path {name!r}")

    def read(self, name: str) -> bytes:
        member = self._members.get(name)
        if member is None:
            raise VerificationError(f"OCI archive {self.path} is missing {name}")
        stream = self._archive.extractfile(member)
        if stream is None:
            raise VerificationError(f"cannot read OCI archive member {name}")
        payload = stream.read()
        if len(payload) != member.size:
            raise VerificationError(f"short read for OCI archive member {name}")
        return payload

    def verify_blob(self, digest: str, expected_size: int | None = None) -> bytes:
        match = SHA256_DIGEST.fullmatch(digest)
        if match is None:
            raise VerificationError(f"invalid OCI descriptor digest {digest!r}")
        name = f"blobs/sha256/{match.group(1)}"
        payload = self.read(name)
        if expected_size is not None and len(payload) != expected_size:
            raise VerificationError(
                f"OCI descriptor size mismatch for {digest}: "
                f"expected {expected_size}, got {len(payload)}"
            )
        if _sha256_bytes(payload) != digest:
            raise VerificationError(f"OCI blob content does not match path digest {digest}")
        return payload

    def blob_digests(self) -> set[str]:
        return {
            f"sha256:{PurePosixPath(name).name}"
            for name in self._members
            if name.startswith("blobs/sha256/")
        }


def _descriptor_digest(
    descriptor: Mapping[str, Any], *, label: str, media_type: str | None = None
) -> tuple[str, int]:
    if media_type is not None and descriptor.get("mediaType") != media_type:
        raise VerificationError(f"{label} has an unexpected media type")
    digest = _required_digest(descriptor, "digest", label)
    size = _required_positive_int(descriptor, "size", label)
    return digest, size


def verify_archive(lock_path: Path, archive_path: Path) -> ArchiveIdentity:
    """Verify one complete OCI archive against the frozen runtime lock."""

    lock = _load_lock(lock_path)
    expected_index = _required_digest(lock, "oci_index_digest", "lock")
    expected_image = _required_digest(lock, "image_digest", "lock")
    expected_config = _required_digest(lock, "config_digest", "lock")

    with _OciArchive(archive_path) as archive:
        layout = _load_json_bytes(archive.read("oci-layout"), "oci-layout")
        if layout != OCI_LAYOUT:
            raise VerificationError("unexpected OCI layout version or fields")

        index_payload = archive.read("index.json")
        index_digest = _sha256_bytes(index_payload)
        if index_digest != expected_index:
            raise VerificationError(
                f"OCI index digest drift: expected {expected_index}, got {index_digest}"
            )
        index = _load_json_bytes(index_payload, "index.json")
        if index.get("schemaVersion") != 2 or index.get("mediaType") != OCI_INDEX_MEDIA_TYPE:
            raise VerificationError("unexpected OCI index schema or media type")
        manifests = index.get("manifests")
        if not isinstance(manifests, list) or len(manifests) != 1:
            raise VerificationError("OCI index must contain exactly one runnable manifest")
        descriptor = manifests[0]
        if not isinstance(descriptor, dict):
            raise VerificationError("OCI image descriptor must be an object")
        image_digest, image_size = _descriptor_digest(
            descriptor, label="index.manifests[0]", media_type=OCI_MANIFEST_MEDIA_TYPE
        )
        if image_digest != expected_image:
            raise VerificationError(
                f"OCI child manifest drift: expected {expected_image}, got {image_digest}"
            )
        platform = _required_mapping(descriptor, "platform", "index.manifests[0]")
        if platform != {"architecture": "amd64", "os": "linux"}:
            raise VerificationError("OCI child platform must be exactly linux/amd64")

        manifest_payload = archive.verify_blob(image_digest, image_size)
        manifest = _load_json_bytes(manifest_payload, "OCI child manifest")
        if (
            manifest.get("schemaVersion") != 2
            or manifest.get("mediaType") != OCI_MANIFEST_MEDIA_TYPE
        ):
            raise VerificationError("unexpected OCI child manifest schema or media type")
        config_descriptor = _required_mapping(manifest, "config", "manifest")
        config_digest, config_size = _descriptor_digest(
            config_descriptor, label="manifest.config", media_type=OCI_CONFIG_MEDIA_TYPE
        )
        if config_digest != expected_config:
            raise VerificationError(
                f"OCI config drift: expected {expected_config}, got {config_digest}"
            )
        config_payload = archive.verify_blob(config_digest, config_size)
        config = _load_json_bytes(config_payload, "OCI image config")
        if config.get("architecture") != "amd64" or config.get("os") != "linux":
            raise VerificationError("OCI config platform must be linux/amd64")

        layers = manifest.get("layers")
        if not isinstance(layers, list) or not layers:
            raise VerificationError("OCI child manifest must contain at least one layer")
        layer_digests: list[str] = []
        for index_value, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise VerificationError(f"manifest.layers[{index_value}] must be an object")
            layer_digest, layer_size = _descriptor_digest(
                layer, label=f"manifest.layers[{index_value}]"
            )
            archive.verify_blob(layer_digest, layer_size)
            layer_digests.append(layer_digest)
        if len(set(layer_digests)) != len(layer_digests):
            raise VerificationError("OCI child manifest contains duplicate layer digests")

        referenced = {image_digest, config_digest, *layer_digests}
        if archive.blob_digests() != referenced:
            raise VerificationError("OCI archive contains missing or unreferenced blobs")

    return ArchiveIdentity(
        index_digest=index_digest,
        image_digest=image_digest,
        config_digest=config_digest,
        layer_digests=tuple(layer_digests),
    )


def verify_archives(lock_path: Path, archive_paths: Sequence[Path]) -> ArchiveIdentity:
    """Verify two independent archives and require their complete identities to agree."""

    if len(archive_paths) != 2:
        raise VerificationError("exactly two independent OCI archives are required")
    if archive_paths[0].resolve() == archive_paths[1].resolve():
        raise VerificationError("independent OCI archive paths must be distinct")
    identities = [verify_archive(lock_path, path) for path in archive_paths]
    if identities[0] != identities[1]:
        raise VerificationError("independent OCI build identities do not match")
    return identities[0]


def _verify_loaded_image_payload(lock: Mapping[str, Any], payload: object) -> None:
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise VerificationError("docker image inspect must return exactly one image")
    image = payload[0]
    expected_reference = _required_string(lock, "image_reference", "lock")
    expected_digest = _required_digest(lock, "image_digest", "lock")
    expected_config = _required_digest(lock, "config_digest", "lock")
    if image.get("Architecture") != "amd64" or image.get("Os") != "linux":
        raise VerificationError("loaded runtime platform must be linux/amd64")
    if image.get("Id") not in {expected_digest, expected_config}:
        raise VerificationError(
            "loaded runtime image ID matches neither the locked manifest nor config"
        )
    repo_digests = image.get("RepoDigests")
    if not isinstance(repo_digests, list) or expected_reference not in repo_digests:
        raise VerificationError("loaded runtime does not expose the exact locked digest reference")
    descriptor = image.get("Descriptor")
    if descriptor is not None and (
        not isinstance(descriptor, dict) or descriptor.get("digest") != expected_digest
    ):
        raise VerificationError("loaded runtime descriptor does not match the locked digest")


def verify_loaded_image(lock_path: Path) -> None:
    """Require Docker to resolve the exact locked digest as a linux/amd64 image."""

    lock = _load_lock(lock_path)
    reference = _required_string(lock, "image_reference", "lock")
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", reference],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise VerificationError(f"cannot invoke Docker to inspect {reference}: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or "no diagnostic"
        raise VerificationError(f"cannot inspect exact locked runtime {reference}: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VerificationError("docker image inspect returned invalid JSON") from exc
    _verify_loaded_image_payload(lock, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        required=True,
        help="path to runtime-image-lock.json",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inputs = subparsers.add_parser("verify-inputs", help="verify the frozen Docker context")
    inputs.add_argument("--context", type=Path, required=True)

    subparsers.add_parser("source-date-epoch", help="print the locked SOURCE_DATE_EPOCH")

    archives = subparsers.add_parser(
        "verify-archives", help="verify two independently built OCI archives"
    )
    archives.add_argument("--archive", action="append", type=Path, required=True)

    subparsers.add_parser("verify-loaded-image", help="verify the exact image loaded in Docker")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify-inputs":
            epoch = verify_build_inputs(args.lock, args.context)
            print(f"verified frozen runtime inputs at SOURCE_DATE_EPOCH={epoch}")
        elif args.command == "source-date-epoch":
            lock = _load_lock(args.lock)
            build = _required_mapping(lock, "build", "lock")
            print(_required_positive_int(build, "source_date_epoch", "lock.build"))
        elif args.command == "verify-archives":
            identity = verify_archives(args.lock, args.archive)
            print(json.dumps(identity.__dict__, sort_keys=True, separators=(",", ":")))
        elif args.command == "verify-loaded-image":
            verify_loaded_image(args.lock)
            print("verified exact loaded linux/amd64 runtime image")
        else:  # pragma: no cover - argparse enforces the closed command set.
            raise VerificationError(f"unsupported command {args.command!r}")
    except VerificationError as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
