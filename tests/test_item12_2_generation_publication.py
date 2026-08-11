"""Source-free crash, coherence, and immutability tests for Item 12.2 publication."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from cellstate.training import publication

ROOT = Path(__file__).resolve().parents[1]


def _outputs(marker: bytes) -> dict[str, bytes]:
    return {
        "backends/component/bundle.json": b'{"marker":"' + marker + b'"}',
        "benchmarks/candidate/model.json": b'{"model":"' + marker + b'"}',
        "benchmarks/candidate/support/runtime.json": b'{"runtime":"' + marker + b'"}',
    }


def _snapshot_marker(snapshot: publication.GenerationSnapshot) -> bytes:
    return snapshot.read_bytes("benchmarks/candidate/model.json")


def test_generations_are_content_addressed_immutable_and_selected_by_one_pointer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "publication"
    first = publication.publish_generation(root, _outputs(b"old"))
    second = publication.publish_generation(root, _outputs(b"new"))

    assert first.pointer.generation_id != second.pointer.generation_id
    assert _snapshot_marker(first) == b'{"model":"old"}'
    assert _snapshot_marker(second) == b'{"model":"new"}'
    assert publication.resolve_current_generation(root).pointer == second.pointer
    assert publication.generation_matches(second, _outputs(b"new"))
    assert not publication.generation_matches(second, _outputs(b"old"))
    assert len(tuple((root / "generations").iterdir())) == 2


def test_pre_render_seed_fixes_generation_before_generation_scoped_uris_exist(
    tmp_path: Path,
) -> None:
    root = tmp_path / "publication"
    seed = b'{"artifact_schema":"pre-render-test","version":"1"}'
    generation_id = publication.generation_id_for_seed(seed)
    outputs = {
        "plan.json": (
            b'{"uri":"https://example.test/generations/'
            + generation_id.encode()
            + b'/tree/model.json"}'
        ),
        "model.json": b'{"model":"frozen"}',
    }
    snapshot = publication.publish_generation(root, outputs, generation_seed=seed)
    assert snapshot.pointer.generation_id == generation_id
    assert snapshot.manifest.generation_id_strategy == "pre_render_seed"
    assert publication.publish_generation(root, outputs, generation_seed=seed).pointer == (
        snapshot.pointer
    )
    with pytest.raises(publication.GenerationPublicationError, match="different file inventory"):
        publication.publish_generation(
            root,
            {**outputs, "model.json": b'{"model":"substituted"}'},
            generation_seed=seed,
        )


def test_concurrent_readers_observe_only_complete_old_or_new_generations(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    publication.publish_generation(root, _outputs(b"old"))
    stop = threading.Event()
    observed: list[bytes] = []
    failures: list[BaseException] = []

    def read_repeatedly() -> None:
        try:
            while not stop.is_set():
                snapshot = publication.resolve_current_generation(root)
                bundle = snapshot.read_bytes("backends/component/bundle.json")
                model = snapshot.read_bytes("benchmarks/candidate/model.json")
                runtime = snapshot.read_bytes("benchmarks/candidate/support/runtime.json")
                marker = b"old" if b"old" in model else b"new"
                assert marker in bundle and marker in runtime
                observed.append(marker)
        except BaseException as error:  # pragma: no cover - asserted below
            failures.append(error)

    reader = threading.Thread(target=read_repeatedly)
    reader.start()
    try:
        publication.publish_generation(root, _outputs(b"new"))
    finally:
        stop.set()
        reader.join(timeout=5)
    assert not reader.is_alive()
    assert failures == []
    assert observed
    assert set(observed) <= {b"old", b"new"}


def _run_crashing_publisher(root: Path, phase: str) -> subprocess.CompletedProcess[str]:
    code = r"""
import os
import signal
import sys
from pathlib import Path
from cellstate.training import publication

root = Path(sys.argv[1])
phase = sys.argv[2]
outputs = {
    "backends/component/bundle.json": b'{"marker":"new"}',
    "benchmarks/candidate/model.json": b'{"model":"new"}',
    "benchmarks/candidate/support/runtime.json": b'{"runtime":"new"}',
}
if phase == "staged":
    original = publication._fsync_tree
    def crash_after_stage(path):
        original(path)
        if path.name == "generation" and path.parent.parent.name == ".staging":
            os.kill(os.getpid(), signal.SIGKILL)
    publication._fsync_tree = crash_after_stage
elif phase == "renamed":
    def crash_after_unsealed_rename(path):
        assert path.parent.name == "generations"
        os.kill(os.getpid(), signal.SIGKILL)
    publication._after_unsealed_generation_rename = crash_after_unsealed_rename
elif phase == "installed":
    original = publication._fsync_directory
    def crash_after_install(path):
        original(path)
        if path.name == "generations":
            os.kill(os.getpid(), signal.SIGKILL)
    publication._fsync_directory = crash_after_install
elif phase == "pointed":
    original = publication._fsync_directory
    root_syncs = 0
    def crash_after_pointer(path):
        global root_syncs
        original(path)
        if path == root:
            root_syncs += 1
            if root_syncs == 2:
                os.kill(os.getpid(), signal.SIGKILL)
    publication._fsync_directory = crash_after_pointer
else:
    raise AssertionError(phase)
publication.publish_generation(root, outputs)
"""
    return subprocess.run(
        [sys.executable, "-c", code, str(root), phase],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        check=False,
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    ("phase", "expected_after_crash"),
    (("staged", b"old"), ("installed", b"old"), ("pointed", b"new")),
)
def test_sigkill_at_each_visibility_boundary_recovers_without_mixed_files(
    tmp_path: Path,
    phase: str,
    expected_after_crash: bytes,
) -> None:
    root = tmp_path / "publication"
    publication.publish_generation(root, _outputs(b"old"))
    crashed = _run_crashing_publisher(root, phase)
    assert crashed.returncode == -signal.SIGKILL

    observed = publication.resolve_current_generation(root)
    assert expected_after_crash in _snapshot_marker(observed)
    recovered = publication.publish_generation(root, _outputs(b"new"))
    assert _snapshot_marker(recovered) == b'{"model":"new"}'
    assert tuple((root / ".staging").iterdir()) == ()
    assert not tuple(root.glob(".current.*.tmp"))


def test_stable_flock_is_released_by_process_death_and_orphans_are_recovered(
    tmp_path: Path,
) -> None:
    root = tmp_path / "publication"
    publication.publish_generation(root, _outputs(b"old"))
    ready = tmp_path / "ready"
    code = r"""
import os
import sys
from pathlib import Path
from cellstate.training import publication
root = Path(sys.argv[1])
ready = Path(sys.argv[2])
with publication._publication_lock(root):
    ready.write_text("locked")
    os.kill(os.getpid(), __import__("signal").SIGSTOP)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(root), str(ready)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    try:
        for _ in range(1_000):
            if ready.exists():
                break
            process.poll()
            time.sleep(0.005)
        assert ready.exists()
        process.kill()
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    orphan = root / ".staging" / ("stage-" + "a" * 32)
    orphan.mkdir()
    (orphan / "partial").write_bytes(b"partial")
    (root / (".current." + "b" * 32 + ".tmp")).write_bytes(b"partial")
    publication.recover_publication(root)
    assert not orphan.exists()
    assert not tuple(root.glob(".current.*.tmp"))
    assert _snapshot_marker(publication.resolve_current_generation(root)) == b'{"model":"old"}'


def test_recovery_reseals_generation_left_writable_between_rename_and_seal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "publication"
    publication.publish_generation(root, _outputs(b"old"))
    crashed = _run_crashing_publisher(root, "renamed")
    assert crashed.returncode == -signal.SIGKILL
    writable = [path for path in (root / "generations").iterdir() if path.stat().st_mode & 0o200]
    assert writable

    recovered = publication.publish_generation(root, _outputs(b"different"))
    assert _snapshot_marker(recovered) == b'{"model":"different"}'
    for generation in (root / "generations").iterdir():
        assert generation.stat().st_mode & 0o777 == 0o555
        for path in generation.rglob("*"):
            expected = 0o555 if path.is_dir() else 0o444
            assert path.stat().st_mode & 0o777 == expected


def test_path_pointer_and_symlink_tampering_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    snapshot = publication.publish_generation(root, _outputs(b"old"))
    with pytest.raises(ValueError, match=r"escape|canonical"):
        publication.publish_generation(root, {"../escape": b"no"})

    artifact = snapshot.generation_root / "tree/benchmarks/candidate/model.json"
    artifact.parent.chmod(0o755)
    artifact.unlink()
    artifact.symlink_to(tmp_path / "outside")
    with pytest.raises(publication.GenerationPublicationError, match="regular file"):
        snapshot.read_bytes("benchmarks/candidate/model.json")
    with pytest.raises(publication.GenerationPublicationError, match=r"symlink|regular file"):
        publication.resolve_current_generation(root)

    pointer = root / "current.json"
    pointer.write_bytes(pointer.read_bytes() + b"\n")
    with pytest.raises(publication.GenerationPublicationError, match="canonical JSON"):
        publication.resolve_current_generation(root)


def test_publication_models_and_output_normalization_reject_ambiguous_identity() -> None:
    with pytest.raises(publication.GenerationPublicationError, match="nonempty"):
        publication.generation_id_for_seed(b"")
    with pytest.raises(publication.GenerationPublicationError, match="not canonical"):
        publication.generation_id_for_seed_sha256("not-a-digest")
    with pytest.raises(ValueError, match="metadata"):
        publication.GenerationEntry(
            relative_path="generation-manifest.json/alias",
            sha256="a" * 64,
            byte_count=1,
        )

    first = publication.GenerationEntry(relative_path="a", sha256="a" * 64, byte_count=1)
    second = publication.GenerationEntry(relative_path="b", sha256="b" * 64, byte_count=1)
    with pytest.raises(ValueError, match="sorted"):
        publication.GenerationManifest(
            generation_id="c" * 64,
            entries=(second, first),
        )
    with pytest.raises(ValueError, match="must not carry"):
        publication.GenerationManifest(
            generation_id=publication._generation_id((first,)),
            generation_seed_sha256="d" * 64,
            entries=(first,),
        )
    with pytest.raises(ValueError, match="seed SHA-256"):
        publication.GenerationManifest(
            generation_id="e" * 64,
            generation_id_strategy="pre_render_seed",
            entries=(first,),
        )
    with pytest.raises(ValueError, match="generation ID"):
        publication.GenerationManifest(generation_id="f" * 64, entries=(first,))

    with pytest.raises(publication.GenerationPublicationError, match="key"):
        publication._normalized_outputs({1: b"bytes"})  # type: ignore[dict-item]
    with pytest.raises(publication.GenerationPublicationError, match="duplicate"):
        publication._normalized_outputs({"same": b"one", Path("same"): b"two"})
    with pytest.raises(publication.GenerationPublicationError, match="exact bytes"):
        publication._normalized_outputs({"model": "text"})  # type: ignore[dict-item]
    with pytest.raises(publication.GenerationPublicationError, match="empty"):
        publication._normalized_outputs({})


def test_generation_verification_rejects_pointer_closure_bytes_and_mode_drift(
    tmp_path: Path,
) -> None:
    def fresh(name: str) -> publication.GenerationSnapshot:
        return publication.publish_generation(tmp_path / name, _outputs(name.encode()))

    wrong_id = fresh("wrong-id")
    with pytest.raises(publication.GenerationPublicationError, match="wrong generation ID"):
        publication.verify_generation(
            wrong_id.generation_root,
            expected_generation_id="f" * 64,
        )
    with pytest.raises(publication.GenerationPublicationError, match="absent"):
        wrong_id.read_bytes("not-declared.json")

    extra = fresh("extra")
    extra.generation_root.chmod(0o755)
    tree = extra.generation_root / "tree"
    tree.chmod(0o755)
    unexpected = tree / "unexpected.json"
    unexpected.write_bytes(b"extra")
    unexpected.chmod(0o444)
    tree.chmod(0o555)
    extra.generation_root.chmod(0o555)
    with pytest.raises(publication.GenerationPublicationError, match="closure drifted"):
        publication.verify_generation(extra.generation_root)

    changed = fresh("changed")
    artifact = changed.generation_root / "tree/benchmarks/candidate/model.json"
    artifact.parent.chmod(0o755)
    artifact.chmod(0o644)
    artifact.write_bytes(b"changed")
    artifact.chmod(0o444)
    artifact.parent.chmod(0o555)
    with pytest.raises(publication.GenerationPublicationError, match="differs from its manifest"):
        publication.verify_generation(changed.generation_root)

    writable_file = fresh("writable-file")
    artifact = writable_file.generation_root / "tree/benchmarks/candidate/model.json"
    artifact.chmod(0o644)
    with pytest.raises(publication.GenerationPublicationError, match="artifact is not sealed"):
        publication.verify_generation(writable_file.generation_root)

    writable_directory = fresh("writable-directory")
    writable_directory.generation_root.chmod(0o755)
    with pytest.raises(publication.GenerationPublicationError, match="directory is not sealed"):
        publication.verify_generation(writable_directory.generation_root)

    wrong_pointer = fresh("wrong-pointer")
    pointer_path = wrong_pointer.publication_root / "current.json"
    pointer_path.chmod(0o644)
    substituted = publication.GenerationPointer(
        generation_id=wrong_pointer.pointer.generation_id,
        manifest_sha256="0" * 64,
        manifest_byte_count=wrong_pointer.pointer.manifest_byte_count,
    )
    pointer_path.write_bytes(publication.canonical_json_bytes(substituted.model_dump(mode="json")))
    pointer_path.chmod(0o444)
    with pytest.raises(publication.GenerationPublicationError, match="does not bind"):
        publication.resolve_current_generation(wrong_pointer.publication_root)


def test_recovery_rejects_unowned_or_malformed_layout_objects(tmp_path: Path) -> None:
    root_file = tmp_path / "root-file"
    root_file.write_bytes(b"not-a-directory")
    with pytest.raises(publication.GenerationPublicationError, match="real directory"):
        publication._validate_publication_root(root_file)

    symlink_layout = tmp_path / "symlink-layout"
    symlink_layout.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (symlink_layout / "generations").symlink_to(outside, target_is_directory=True)
    with pytest.raises(publication.GenerationPublicationError, match="layout path"):
        publication._prepare_layout(symlink_layout)

    valid_root = tmp_path / "valid"
    publication._prepare_layout(valid_root)
    with pytest.raises(publication.GenerationPublicationError, match="unowned staging"):
        publication._safe_remove_stage(
            valid_root / ".staging/not-owned",
            staging_root=valid_root / ".staging",
        )
    absent_stage = valid_root / ".staging" / ("stage-" + "a" * 32)
    publication._safe_remove_stage(absent_stage, staging_root=valid_root / ".staging")

    unknown_stage_root = tmp_path / "unknown-stage"
    publication._prepare_layout(unknown_stage_root)
    (unknown_stage_root / ".staging/unknown").write_bytes(b"unknown")
    with pytest.raises(publication.GenerationPublicationError, match="unknown object"):
        publication.recover_publication(unknown_stage_root)

    pointer_temp_root = tmp_path / "pointer-temp"
    publication._prepare_layout(pointer_temp_root)
    (pointer_temp_root / (".current." + "b" * 32 + ".tmp")).mkdir()
    with pytest.raises(publication.GenerationPublicationError, match="pointer temporary"):
        publication.recover_publication(pointer_temp_root)

    unknown_generation_root = tmp_path / "unknown-generation"
    publication._prepare_layout(unknown_generation_root)
    (unknown_generation_root / "generations/unknown").mkdir()
    with pytest.raises(publication.GenerationPublicationError, match="generation store"):
        publication.recover_publication(unknown_generation_root)

    non_directory_generation_root = tmp_path / "non-directory-generation"
    publication._prepare_layout(non_directory_generation_root)
    (non_directory_generation_root / "generations" / ("c" * 64)).write_bytes(b"not-a-tree")
    with pytest.raises(publication.GenerationPublicationError, match="not a real directory"):
        publication.recover_publication(non_directory_generation_root)
