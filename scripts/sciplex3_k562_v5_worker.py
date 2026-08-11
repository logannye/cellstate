#!/usr/bin/env python3
"""Contained sci-Plex3 v5 worker; stages artifacts and never publishes them."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path


def _hash_descriptor(descriptor: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError("source snapshot descriptor is not a regular file")
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        byte_count += len(chunk)
    after = os.fstat(descriptor)
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise RuntimeError("source snapshot changed while being authenticated")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), byte_count


def _snapshot_pinned_source(
    source: Path,
    snapshot_directory: Path,
    *,
    snapshot_max_bytes: int,
) -> tuple[int, str, int]:
    """Copy the Docker-pinned bind once, then hold an unlinked immutable private inode."""

    read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(source, read_flags)
    snapshot_path = snapshot_directory / ".contained-source-snapshot"
    write_flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    snapshot_descriptor = -1
    try:
        source_before = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_before.st_mode):
            raise RuntimeError("Docker source bind is not one regular file")
        if source_before.st_size > snapshot_max_bytes:
            raise RuntimeError("Docker source bind exceeds the bounded snapshot volume budget")
        snapshot_descriptor = os.open(snapshot_path, write_flags, 0o600)
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            if byte_count > snapshot_max_bytes:
                raise RuntimeError("source snapshot crossed its exact volume budget")
            offset = 0
            while offset < len(chunk):
                written = os.write(snapshot_descriptor, chunk[offset:])
                if written <= 0:
                    raise RuntimeError("short private source-snapshot write")
                offset += written
        source_after = os.fstat(source_descriptor)
        if (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
        ) != (source_after.st_dev, source_after.st_ino, source_after.st_size):
            raise RuntimeError("Docker source bind changed while snapshotting")
        os.fsync(snapshot_descriptor)
        os.fchmod(snapshot_descriptor, 0o400)
        read_descriptor = os.open(snapshot_path, read_flags)
        os.close(snapshot_descriptor)
        snapshot_descriptor = -1
        snapshot_path.unlink()
        return read_descriptor, digest.hexdigest(), byte_count
    except BaseException:
        if snapshot_descriptor >= 0:
            os.close(snapshot_descriptor)
        with suppress(FileNotFoundError):
            snapshot_path.unlink()
        raise
    finally:
        os.close(source_descriptor)


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        if os.write(descriptor, payload) != len(payload):
            raise RuntimeError("short contained-worker observation write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_worker(
    *,
    source_path: Path,
    output_path: Path,
    repository_root: Path,
    execution_id: str,
    expected_source_sha256: str,
    expected_source_byte_count: int,
    snapshot_directory: Path,
    snapshot_max_bytes: int,
) -> None:
    """Authenticate the pinned bind, fit v5, reauthenticate, and leave only a stage."""

    snapshot_descriptor, pre_sha256, pre_byte_count = _snapshot_pinned_source(
        source_path,
        snapshot_directory,
        snapshot_max_bytes=snapshot_max_bytes,
    )
    if (pre_sha256, pre_byte_count) != (expected_source_sha256, expected_source_byte_count):
        os.close(snapshot_descriptor)
        raise RuntimeError("pinned source bytes differ before contained training")

    # Import every source-touching or numerical dependency only after the source is pinned and
    # authenticated inside Docker's aggregate cgroup.
    import scripts.materialize_sciplex3_k562_p1_candidate as materializer
    from cellstate.domain.common import canonical_json_bytes
    from cellstate.evaluation import sciplex3_candidate_runner as runner
    from cellstate.training.execution import (
        ContainedTrainingWorkerObservation,
        inventory_staged_training_tree,
    )

    policy, code_closure, input_closure, runtime_image_lock = runner.contained_training_contracts(
        repository_root
    )
    snapshot_proc_path = Path(f"/proc/self/fd/{snapshot_descriptor}")
    try:
        preparation = materializer._prepare_exact_p1(snapshot_proc_path, repository_root)
        repository_bindings = materializer._repository_bindings(repository_root)
        support_fingerprint, _ = materializer._planned_support_envelope(repository_root)
        plan = runner.build_sciplex3_candidate_training_plan(
            preparation,
            benchmark_fingerprint=materializer._binding_sha256(repository_bindings, "benchmark"),
            support_envelope_fingerprint=support_fingerprint,
        )

        sealed = runner.seal_sciplex3_candidate_training_plan(
            preparation, plan, output_path / "sealed-plan"
        )
        fitted = runner.fit_and_write_sciplex3_candidate(preparation, sealed, output_path / "fit")
        observation = runner.verify_sciplex3_candidate_fit(preparation, sealed, fitted)
        post_sha256, post_byte_count = _hash_descriptor(snapshot_descriptor)
    finally:
        os.close(snapshot_descriptor)
    shutil.copyfile(sealed.artifact.path, output_path / "candidate-training-plan.json")
    shutil.copyfile(fitted.model_artifact.path, output_path / "candidate-model.json")
    shutil.copyfile(
        fitted.observation_artifact.path,
        output_path / "training-execution-observation.json",
    )
    (output_path / "p1-finalized-count-scan-receipt.json").write_bytes(
        canonical_json_bytes(preparation.finalized_count_scan_manifest())
    )
    (output_path / "p1-assembly-receipt.json").write_bytes(
        canonical_json_bytes(asdict(preparation.receipt))
    )

    sealed_support = {artifact.path.name: artifact.path for artifact in sealed.support_artifacts}
    required_legacy_support = {
        "candidate-specification.json",
        "output-model-schema.json",
        "p1-count-stream-descriptor.json",
        "runtime-lock.json",
    }
    if not required_legacy_support.issubset(sealed_support):
        raise RuntimeError("sealed contained plan lacks the materialization support closure")
    vertical = materializer.BENCHMARK_RELATIVE_DIRECTORY / "item12-p1"
    artifact_payloads = {
        "assembly_receipt": (
            vertical / "p1-assembly-receipt.json",
            (output_path / "p1-assembly-receipt.json").read_bytes(),
        ),
        "candidate_model": (
            vertical / "candidate-model.json",
            (output_path / "candidate-model.json").read_bytes(),
        ),
        "candidate_output_model_schema": (
            materializer.SUPPORT_RELATIVE_PATHS["output-model-schema.json"],
            sealed_support["output-model-schema.json"].read_bytes(),
        ),
        "candidate_runtime_lock": (
            materializer.SUPPORT_RELATIVE_PATHS["runtime-lock.json"],
            sealed_support["runtime-lock.json"].read_bytes(),
        ),
        "candidate_specification": (
            materializer.SUPPORT_RELATIVE_PATHS["candidate-specification.json"],
            sealed_support["candidate-specification.json"].read_bytes(),
        ),
        "candidate_training_plan": (
            vertical / "candidate-training-plan.json",
            (output_path / "candidate-training-plan.json").read_bytes(),
        ),
        "finalized_count_scan_receipt": (
            vertical / "p1-finalized-count-scan-receipt.json",
            (output_path / "p1-finalized-count-scan-receipt.json").read_bytes(),
        ),
        "p1_count_stream_descriptor": (
            materializer.COUNT_DESCRIPTOR_RELATIVE_PATH,
            sealed_support["p1-count-stream-descriptor.json"].read_bytes(),
        ),
        "training_execution_observation": (
            vertical / "training-execution-observation.json",
            (output_path / "training-execution-observation.json").read_bytes(),
        ),
    }
    materialization = materializer._build_manifest(
        preparation,
        plan,
        observation,
        artifact_payloads,
        repository_bindings,
        support_envelope_fingerprint=support_fingerprint,
    )
    _write_exclusive(
        output_path / "materialization-manifest.json",
        canonical_json_bytes(materialization),
    )

    if (post_sha256, post_byte_count) != (expected_source_sha256, expected_source_byte_count):
        raise RuntimeError("private pinned source snapshot differs after contained training")
    staged_inventory = inventory_staged_training_tree(output_path)
    observation = ContainedTrainingWorkerObservation(
        execution_id=execution_id,
        training_plan_fingerprint=plan.fingerprint,
        policy_fingerprint=policy.fingerprint,
        runtime_image_digest=runtime_image_lock.runtime_image.digest,
        training_code_closure_sha256=code_closure.fingerprint,
        execution_input_closure_sha256=input_closure.fingerprint,
        expected_source_sha256=expected_source_sha256,
        source_pre_sha256=pre_sha256,
        source_post_sha256=post_sha256,
        expected_source_byte_count=expected_source_byte_count,
        source_pre_byte_count=pre_byte_count,
        source_post_byte_count=post_byte_count,
        staged_inventory=staged_inventory,
        staged_tree_sha256=staged_inventory.fingerprint,
        staged_file_count=len(staged_inventory.entries),
    )
    _write_exclusive(
        output_path / "contained-worker-observation.json",
        canonical_json_bytes(observation.model_dump(mode="json")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-source-byte-count", type=int, required=True)
    parser.add_argument("--snapshot-directory", type=Path, required=True)
    parser.add_argument("--snapshot-max-bytes", type=int, required=True)
    args = parser.parse_args()
    run_worker(
        source_path=args.source,
        output_path=args.output,
        repository_root=args.repository_root,
        execution_id=args.execution_id,
        expected_source_sha256=args.expected_source_sha256,
        expected_source_byte_count=args.expected_source_byte_count,
        snapshot_directory=args.snapshot_directory,
        snapshot_max_bytes=args.snapshot_max_bytes,
    )


if __name__ == "__main__":
    main()
