#!/usr/bin/env python3
"""Host supervisor for a source-free-controlled, stage-only sci-Plex3 v5 fit."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from cellstate.domain.common import canonical_json_bytes
from cellstate.evaluation.sciplex3_candidate_runner import contained_training_contracts
from cellstate.training.execution import (
    ContainedExecutionError,
    ContainedTrainingObservation,
    ContainedTrainingWorkerObservation,
    DockerExecutor,
    inventory_staged_training_tree,
    seal_staged_training_tree,
)


def _read_canonical_worker_observation(path: Path) -> ContainedTrainingWorkerObservation:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        expected_size = os.fstat(descriptor).st_size
        payload = os.read(descriptor, expected_size + 1)
    finally:
        os.close(descriptor)
    observation = ContainedTrainingWorkerObservation.model_validate_json(payload)
    if canonical_json_bytes(observation.model_dump(mode="json")) != payload:
        raise ContainedExecutionError("worker observation is not canonical JSON")
    return observation


def run_contained_training(
    *,
    source_path: Path,
    repository_root: Path,
    staging_root: Path,
    canonical_publication_root: Path,
    execution_id: str,
) -> tuple[Path, ContainedTrainingObservation]:
    """Run and verify one stage; this function never opens source bytes or publishes output."""

    policy, code_closure, input_closure, image_lock = contained_training_contracts(repository_root)
    executor = DockerExecutor(
        policy,
        lock_root=staging_root / ".locks",
        staging_root=staging_root,
        canonical_publication_root=canonical_publication_root,
        execution_input_closure=input_closure,
    )
    output = executor.output_stage_path(execution_id)
    execution = executor.run(
        execution_id=execution_id,
        source_path=source_path,
        code_path=repository_root,
        output_path=output,
    )
    worker = _read_canonical_worker_observation(output / "contained-worker-observation.json")
    staged_inventory = inventory_staged_training_tree(output)
    if staged_inventory != worker.staged_inventory:
        raise ContainedExecutionError("worker stage differs on parent no-follow re-inventory")
    observation = ContainedTrainingObservation(
        training_plan_fingerprint=worker.training_plan_fingerprint,
        policy_fingerprint=policy.fingerprint,
        runtime_image_digest=image_lock.runtime_image.digest,
        training_code_closure_sha256=code_closure.fingerprint,
        execution_input_closure_sha256=input_closure.fingerprint,
        staged_inventory=staged_inventory,
        staged_tree_sha256=staged_inventory.fingerprint,
        worker_observation=worker,
        execution_observation=execution,
    )
    path = output / "contained-training-observation.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        payload = canonical_json_bytes(observation.model_dump(mode="json"))
        if os.write(descriptor, payload) != len(payload):
            raise ContainedExecutionError("short parent observation write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    seal_staged_training_tree(output, expected_inventory=staged_inventory)
    return output, observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5ad", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--canonical-publication-root", type=Path, required=True)
    parser.add_argument("--execution-id", required=True)
    args = parser.parse_args()
    output, observation = run_contained_training(
        source_path=args.source_h5ad,
        repository_root=args.repository_root,
        staging_root=args.staging_root,
        canonical_publication_root=args.canonical_publication_root,
        execution_id=args.execution_id,
    )
    print(f"contained_stage {output}")
    print(f"contained_training_observation_sha256 {observation.fingerprint}")


if __name__ == "__main__":
    main()
