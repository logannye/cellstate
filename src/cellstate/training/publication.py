"""Crash-safe publication of immutable, content-addressed artifact generations.

The only mutable reader-visible object is ``current.json``.  A writer first builds and
verifies a complete generation on the same filesystem, renames that directory into the
immutable generation store, and only then atomically replaces the pointer.  Readers take one
pointer snapshot and never combine paths from different generations.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator, model_validator

from cellstate.domain.common import SchemaModel, canonical_fingerprint, canonical_json_bytes

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STAGE_PATTERN = re.compile(r"^stage-[0-9a-f]{32}$")
_POINTER_TEMP_PATTERN = re.compile(r"^\.current\.[0-9a-f]{32}\.tmp$")
_MANIFEST_FILENAME = "generation-manifest.json"
_POINTER_FILENAME = "current.json"
_TREE_DIRECTORY = "tree"
_GENERATIONS_DIRECTORY = "generations"
_STAGING_DIRECTORY = ".staging"
_LOCK_FILENAME = ".publication.lock"


class GenerationPublicationError(RuntimeError):
    """Raised when generation storage cannot prove one coherent immutable snapshot."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_relative_path(value: str) -> str:
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        raise ValueError("generation entry path must be a nonempty canonical POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("generation entry path must not escape or alias its generation")
    if path.parts[0] in {_MANIFEST_FILENAME, _POINTER_FILENAME}:
        raise ValueError("generation entry path collides with publication metadata")
    return value


class GenerationEntry(SchemaModel):
    """One exact logical path inside a generation tree."""

    model_config = ConfigDict(strict=True)

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def path_is_canonical(cls, value: str) -> str:
        return _canonical_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def digest_is_canonical(cls, value: str) -> str:
        if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("generation entry SHA-256 must be lowercase canonical hex")
        return value

    @field_validator("byte_count")
    @classmethod
    def byte_count_is_exact(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("generation entry byte count must be an exact integer")
        return value


def _generation_id(entries: Sequence[GenerationEntry]) -> str:
    return canonical_fingerprint(
        {
            "artifact_schema": "cellstate-immutable-generation-seed",
            "artifact_schema_version": "1.0.0",
            "entries": [entry.model_dump(mode="json") for entry in entries],
        }
    )


def generation_id_for_seed(generation_seed: bytes) -> str:
    """Derive an ID before rendering payloads that embed their immutable generation URI."""

    if type(generation_seed) is not bytes or not generation_seed:
        raise GenerationPublicationError("generation seed must be nonempty exact bytes")
    return generation_id_for_seed_sha256(_sha256(generation_seed))


def generation_id_for_seed_sha256(generation_seed_sha256: str) -> str:
    """Derive the pre-render generation ID from an already authenticated seed digest."""

    if (
        type(generation_seed_sha256) is not str
        or _SHA256_PATTERN.fullmatch(generation_seed_sha256) is None
    ):
        raise GenerationPublicationError("generation seed SHA-256 is not canonical")
    return canonical_fingerprint(
        {
            "artifact_schema": "cellstate-immutable-generation-seed",
            "artifact_schema_version": "2.0.0",
            "generation_seed_sha256": generation_seed_sha256,
        }
    )


class GenerationManifest(SchemaModel):
    """Closed file inventory for one immutable generation."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-immutable-generation"] = "cellstate-immutable-generation"
    artifact_schema_version: Literal["1.0.0"] = "1.0.0"
    generation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_id_strategy: Literal["inventory", "pre_render_seed"] = "inventory"
    generation_seed_sha256: str | None = None
    entries: tuple[GenerationEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def inventory_is_canonical(self) -> Self:
        paths = tuple(entry.relative_path for entry in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("generation entries must be unique and canonically sorted")
        if self.generation_id_strategy == "inventory":
            expected_generation_id = _generation_id(self.entries)
            if self.generation_seed_sha256 is not None:
                raise ValueError("inventory generation must not carry a pre-render seed")
        else:
            if (
                type(self.generation_seed_sha256) is not str
                or _SHA256_PATTERN.fullmatch(self.generation_seed_sha256) is None
            ):
                raise ValueError("pre-render generation seed SHA-256 is not canonical")
            expected_generation_id = generation_id_for_seed_sha256(self.generation_seed_sha256)
        if self.generation_id != expected_generation_id:
            raise ValueError("generation ID does not match its exact file inventory")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


class GenerationPointer(SchemaModel):
    """The single atomically replaced reader-visible generation selector."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-current-generation"] = "cellstate-current-generation"
    artifact_schema_version: Literal["1.0.0"] = "1.0.0"
    generation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_byte_count: int = Field(gt=0)

    @field_validator("generation_id", "manifest_sha256")
    @classmethod
    def digests_are_canonical(cls, value: str) -> str:
        if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("generation pointer digests must be lowercase canonical hex")
        return value

    @field_validator("manifest_byte_count")
    @classmethod
    def byte_count_is_exact(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("generation manifest byte count must be an exact integer")
        return value


def _parse_canonical_model(payload: bytes, model: type[SchemaModel], *, name: str) -> SchemaModel:
    try:
        parsed = model.model_validate_json(payload)
    except ValueError as error:
        raise GenerationPublicationError(f"invalid {name}") from error
    if canonical_json_bytes(parsed.model_dump(mode="json")) != payload:
        raise GenerationPublicationError(f"{name} is not canonical JSON")
    return parsed


def _read_regular_file(path: Path, *, name: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise GenerationPublicationError(f"cannot inspect {name}: {path}") from error
    if not stat.S_ISREG(before.st_mode):
        raise GenerationPublicationError(f"{name} is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            observed = os.fstat(descriptor)
            if (observed.st_dev, observed.st_ino) != (before.st_dev, before.st_ino):
                raise GenerationPublicationError(f"{name} changed while opening: {path}")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read()
        finally:
            os.close(descriptor)
    except OSError as error:
        raise GenerationPublicationError(f"cannot read {name}: {path}") from error
    return payload


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, mode)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                written = handle.write(payload)
                handle.flush()
                os.fsync(descriptor)
            if written != len(payload):
                raise GenerationPublicationError(f"short write while staging {path}")
        finally:
            os.close(descriptor)
    except OSError as error:
        raise GenerationPublicationError(f"cannot exclusively stage {path}") from error


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise GenerationPublicationError(f"cannot durably synchronize directory: {path}") from error


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for current, names, _ in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in names:
            child = current_path / name
            if child.is_symlink() or not child.is_dir():
                raise GenerationPublicationError(
                    f"generation contains a non-directory link: {child}"
                )
            directories.append(child)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _validate_publication_root(path: Path) -> None:
    try:
        observed = path.lstat()
    except OSError as error:
        raise GenerationPublicationError(f"cannot inspect publication root: {path}") from error
    if not stat.S_ISDIR(observed.st_mode):
        raise GenerationPublicationError("publication root must be a real directory")


def _prepare_layout(publication_root: Path) -> None:
    publication_root.mkdir(parents=True, exist_ok=True)
    _validate_publication_root(publication_root)
    devices = {publication_root.lstat().st_dev}
    for name in (_GENERATIONS_DIRECTORY, _STAGING_DIRECTORY):
        path = publication_root / name
        path.mkdir(exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise GenerationPublicationError(f"publication layout path is not a directory: {path}")
        devices.add(path.lstat().st_dev)
    if len(devices) != 1:
        raise GenerationPublicationError(
            "staging, generations, and pointer must share one atomic-rename filesystem"
        )


@contextmanager
def _publication_lock(publication_root: Path) -> Iterator[None]:
    """Hold a stable inode lock; never unlinking it avoids stale-inode races."""

    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - v5 execution is POSIX-only
        raise GenerationPublicationError("generation publication requires POSIX flock") from error
    lock_path = publication_root / _LOCK_FILENAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
    except OSError as error:
        raise GenerationPublicationError("cannot acquire stable publication lock") from error


def _safe_remove_stage(path: Path, *, staging_root: Path) -> None:
    if path.parent != staging_root or _STAGE_PATTERN.fullmatch(path.name) is None:
        raise GenerationPublicationError(f"refusing to remove an unowned staging path: {path}")
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(observed.st_mode):
        raise GenerationPublicationError(f"orphan staging path is not a real directory: {path}")
    # A crash may leave a completely sealed candidate in staging.  Make only real directories
    # inside this UUID-owned stage writable enough to unlink their entries; never follow links.
    directories = [path]
    for current, names, _ in os.walk(path, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in names:
            child = current_path / name
            child_state = child.lstat()
            if stat.S_ISDIR(child_state.st_mode):
                directories.append(child)
    for directory in directories:
        directory.chmod(0o700)
    shutil.rmtree(path)


def _recover_locked(publication_root: Path) -> None:
    staging_root = publication_root / _STAGING_DIRECTORY
    for child in tuple(staging_root.iterdir()):
        if _STAGE_PATTERN.fullmatch(child.name) is None:
            raise GenerationPublicationError(f"unknown object in generation staging area: {child}")
        _safe_remove_stage(child, staging_root=staging_root)
    for child in tuple(publication_root.iterdir()):
        if _POINTER_TEMP_PATTERN.fullmatch(child.name) is None:
            continue
        observed = child.lstat()
        if not stat.S_ISREG(observed.st_mode):
            raise GenerationPublicationError(f"pointer temporary is not a regular file: {child}")
        child.unlink()
    generations_root = publication_root / _GENERATIONS_DIRECTORY
    for child in tuple(generations_root.iterdir()):
        if _SHA256_PATTERN.fullmatch(child.name) is None:
            raise GenerationPublicationError(f"unknown object in generation store: {child}")
        observed = child.lstat()
        if not stat.S_ISDIR(observed.st_mode):
            raise GenerationPublicationError(
                f"installed generation is not a real directory: {child}"
            )
        # A process can die in the platform-required root-mode rename window.  Reseal every
        # canonical installed directory before any later publication; malformed inventories
        # remain fail-closed rather than being silently blessed or pointed.
        _seal_generation(child)
        verify_generation(child, expected_generation_id=child.name)
    _fsync_directory(staging_root)
    _fsync_directory(generations_root)
    _fsync_directory(publication_root)


def _after_unsealed_generation_rename(_installed: Path) -> None:
    """Test hook for the only platform-required rename-to-reseal crash window."""


def recover_publication(publication_root: Path) -> None:
    """Remove only provably orphaned stages and pointer temporaries under the writer lock."""

    root = Path(publication_root)
    _prepare_layout(root)
    with _publication_lock(root):
        _recover_locked(root)


def _expected_directories(entries: Sequence[GenerationEntry]) -> set[str]:
    expected = {_TREE_DIRECTORY}
    for entry in entries:
        parent = PurePosixPath(_TREE_DIRECTORY, entry.relative_path).parent
        while parent.as_posix() not in {".", _TREE_DIRECTORY}:
            expected.add(parent.as_posix())
            parent = parent.parent
    return expected


def verify_generation(
    generation_directory: Path,
    *,
    expected_generation_id: str | None = None,
) -> GenerationManifest:
    """Verify exact bytes, paths, and directory closure for one immutable generation."""

    directory = Path(generation_directory)
    if directory.is_symlink() or not directory.is_dir():
        raise GenerationPublicationError("generation path must be a real directory")
    manifest_payload = _read_regular_file(
        directory / _MANIFEST_FILENAME, name="generation manifest"
    )
    manifest = _parse_canonical_model(
        manifest_payload, GenerationManifest, name="generation manifest"
    )
    assert isinstance(manifest, GenerationManifest)
    if expected_generation_id is not None and manifest.generation_id != expected_generation_id:
        raise GenerationPublicationError("generation directory has the wrong generation ID")

    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for current, directory_names, file_names in os.walk(directory, followlinks=False):
        current_path = Path(current)
        current_relative = current_path.relative_to(directory).as_posix()
        if current_relative != ".":
            actual_directories.add(current_relative)
        for name in directory_names:
            child = current_path / name
            if child.is_symlink() or not child.is_dir():
                raise GenerationPublicationError(
                    f"generation contains a directory symlink: {child}"
                )
        for name in file_names:
            child = current_path / name
            relative = child.relative_to(directory).as_posix()
            _read_regular_file(child, name="generation artifact")
            actual_files.add(relative)

    expected_files = {_MANIFEST_FILENAME} | {
        PurePosixPath(_TREE_DIRECTORY, entry.relative_path).as_posix() for entry in manifest.entries
    }
    if actual_files != expected_files or actual_directories != _expected_directories(
        manifest.entries
    ):
        raise GenerationPublicationError("generation file or directory closure drifted")
    for entry in manifest.entries:
        payload = _read_regular_file(
            directory / _TREE_DIRECTORY / Path(entry.relative_path),
            name=f"generation entry {entry.relative_path}",
        )
        if len(payload) != entry.byte_count or _sha256(payload) != entry.sha256:
            raise GenerationPublicationError(
                f"generation entry differs from its manifest: {entry.relative_path}"
            )
    for relative in expected_files:
        path = directory / relative
        observed = path.lstat()
        if stat.S_IMODE(observed.st_mode) != 0o444:
            raise GenerationPublicationError("generation artifact is not sealed read-only")
    for relative in {".", *actual_directories}:
        path = directory if relative == "." else directory / relative
        if stat.S_IMODE(path.lstat().st_mode) != 0o555:
            raise GenerationPublicationError("generation directory is not sealed read-only")
    return manifest


@dataclass(frozen=True, slots=True)
class GenerationSnapshot:
    """One pointer snapshot bound to a fully verified immutable generation."""

    publication_root: Path
    generation_root: Path
    pointer: GenerationPointer
    manifest: GenerationManifest

    def read_bytes(self, relative_path: str | Path) -> bytes:
        canonical = _canonical_relative_path(PurePosixPath(relative_path).as_posix())
        by_path = {entry.relative_path: entry for entry in self.manifest.entries}
        entry = by_path.get(canonical)
        if entry is None:
            raise GenerationPublicationError(f"path is absent from generation: {canonical}")
        payload = _read_regular_file(
            self.generation_root / _TREE_DIRECTORY / Path(canonical),
            name=f"generation snapshot entry {canonical}",
        )
        if len(payload) != entry.byte_count or _sha256(payload) != entry.sha256:
            raise GenerationPublicationError(f"generation changed after snapshot: {canonical}")
        return payload


def resolve_current_generation(publication_root: Path) -> GenerationSnapshot:
    """Read the atomic pointer once, then verify and return only that generation."""

    root = Path(publication_root)
    _validate_publication_root(root)
    pointer_payload = _read_regular_file(root / _POINTER_FILENAME, name="generation pointer")
    pointer = _parse_canonical_model(pointer_payload, GenerationPointer, name="generation pointer")
    assert isinstance(pointer, GenerationPointer)
    generation_root = root / _GENERATIONS_DIRECTORY / pointer.generation_id
    manifest_payload = _read_regular_file(
        generation_root / _MANIFEST_FILENAME, name="pointed generation manifest"
    )
    if (
        len(manifest_payload) != pointer.manifest_byte_count
        or _sha256(manifest_payload) != pointer.manifest_sha256
    ):
        raise GenerationPublicationError("generation pointer does not bind its manifest")
    manifest = verify_generation(generation_root, expected_generation_id=pointer.generation_id)
    return GenerationSnapshot(root, generation_root, pointer, manifest)


def _normalized_outputs(outputs: Mapping[str | Path, bytes]) -> dict[str, bytes]:
    normalized: dict[str, bytes] = {}
    for raw_path, payload in outputs.items():
        if not isinstance(raw_path, (str, Path)):
            raise GenerationPublicationError("generation output key must be a string or Path")
        path = _canonical_relative_path(PurePosixPath(raw_path).as_posix())
        if path in normalized:
            raise GenerationPublicationError(f"duplicate generation output path: {path}")
        if type(payload) is not bytes:
            raise GenerationPublicationError(f"generation payload is not exact bytes: {path}")
        normalized[path] = payload
    if not normalized:
        raise GenerationPublicationError("cannot publish an empty generation")
    return {path: normalized[path] for path in sorted(normalized)}


def _entries_for(outputs: Mapping[str, bytes]) -> tuple[GenerationEntry, ...]:
    entries: list[GenerationEntry] = []
    for path, payload in outputs.items():
        entries.append(
            GenerationEntry(relative_path=path, sha256=_sha256(payload), byte_count=len(payload))
        )
    return tuple(entries)


def _seal_generation(directory: Path) -> None:
    files: list[Path] = []
    directories: list[Path] = [directory]
    for current, names, filenames in os.walk(directory, followlinks=False):
        current_path = Path(current)
        directories.extend(current_path / name for name in names)
        files.extend(current_path / name for name in filenames)
    for path in files:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise GenerationPublicationError(f"generation artifact is not regular: {path}")
        path.chmod(0o444)
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        if not stat.S_ISDIR(path.lstat().st_mode):
            raise GenerationPublicationError(f"generation directory is not real: {path}")
        path.chmod(0o555)
        _fsync_directory(path)


def publish_generation(
    publication_root: Path,
    outputs: Mapping[str | Path, bytes],
    *,
    generation_seed: bytes | None = None,
) -> GenerationSnapshot:
    """Publish a complete immutable generation through one atomic pointer replacement."""

    root = Path(publication_root)
    normalized_outputs = _normalized_outputs(outputs)
    entries = _entries_for(normalized_outputs)
    if generation_seed is None:
        manifest = GenerationManifest(generation_id=_generation_id(entries), entries=entries)
    else:
        generation_id = generation_id_for_seed(generation_seed)
        manifest = GenerationManifest(
            generation_id=generation_id,
            generation_id_strategy="pre_render_seed",
            generation_seed_sha256=_sha256(generation_seed),
            entries=entries,
        )
    manifest_payload = canonical_json_bytes(manifest.model_dump(mode="json"))
    pointer = GenerationPointer(
        generation_id=manifest.generation_id,
        manifest_sha256=_sha256(manifest_payload),
        manifest_byte_count=len(manifest_payload),
    )
    pointer_payload = canonical_json_bytes(pointer.model_dump(mode="json"))
    _prepare_layout(root)

    stage: Path | None = None
    with _publication_lock(root):
        _recover_locked(root)
        staging_root = root / _STAGING_DIRECTORY
        generations_root = root / _GENERATIONS_DIRECTORY
        stage = staging_root / f"stage-{uuid4().hex}"
        stage.mkdir(mode=0o700)
        try:
            candidate = stage / "generation"
            candidate.mkdir(mode=0o700)
            tree = candidate / _TREE_DIRECTORY
            tree.mkdir()
            for entry in entries:
                _write_exclusive(
                    tree / Path(entry.relative_path),
                    normalized_outputs[entry.relative_path],
                )
            _write_exclusive(candidate / _MANIFEST_FILENAME, manifest_payload)
            _fsync_tree(candidate)
            _seal_generation(candidate)
            verified = verify_generation(candidate, expected_generation_id=manifest.generation_id)
            if verified != manifest:
                raise GenerationPublicationError("staged generation changed during verification")
            installed = generations_root / manifest.generation_id
            try:
                installed.lstat()
                installed_present = True
            except FileNotFoundError:
                installed_present = False
            if installed_present:
                # A process death in the platform-required rename/chmod window can leave only
                # the generation root writable; resealing is idempotent and never changes bytes.
                _seal_generation(installed)
                existing = verify_generation(
                    installed, expected_generation_id=manifest.generation_id
                )
                if existing != manifest:
                    raise GenerationPublicationError(
                        "existing generation ID names a different file inventory"
                    )
                _safe_remove_stage(stage, staging_root=staging_root)
                stage = None
            else:
                # macOS refuses to rename a mode-0555 directory even when both parents are
                # writable.  Its descendants are already sealed and verified; make only this
                # unpointed root temporarily writable for rename, then reseal it before the
                # generation-store fsync and long before the current pointer can change.
                candidate.chmod(0o700)
                candidate.replace(installed)
                _after_unsealed_generation_rename(installed)
                _seal_generation(installed)
                _fsync_directory(generations_root)
                _safe_remove_stage(stage, staging_root=staging_root)
                stage = None
            if (
                verify_generation(installed, expected_generation_id=manifest.generation_id)
                != manifest
            ):
                raise GenerationPublicationError(
                    "installed generation failed exact re-verification"
                )

            pointer_temporary = root / f".current.{uuid4().hex}.tmp"
            try:
                _write_exclusive(pointer_temporary, pointer_payload)
                pointer_temporary.replace(root / _POINTER_FILENAME)
                _fsync_directory(root)
            finally:
                pointer_temporary.unlink(missing_ok=True)
        finally:
            if stage is not None and stage.exists():
                _safe_remove_stage(stage, staging_root=root / _STAGING_DIRECTORY)
    return resolve_current_generation(root)


def generation_matches(
    snapshot: GenerationSnapshot,
    outputs: Mapping[str | Path, bytes],
) -> bool:
    """Return whether a snapshot is exactly the supplied logical output mapping."""

    normalized_outputs = _normalized_outputs(outputs)
    entries = _entries_for(normalized_outputs)
    if snapshot.manifest.entries != entries:
        return False
    return all(
        snapshot.read_bytes(entry.relative_path) == normalized_outputs[entry.relative_path]
        for entry in entries
    )


__all__ = [
    "GenerationEntry",
    "GenerationManifest",
    "GenerationPointer",
    "GenerationPublicationError",
    "GenerationSnapshot",
    "generation_id_for_seed",
    "generation_id_for_seed_sha256",
    "generation_matches",
    "publish_generation",
    "recover_publication",
    "resolve_current_generation",
    "verify_generation",
]
