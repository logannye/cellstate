"""Deterministic source-free tests for Item 12.2 Docker process-tree containment."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from cellstate.domain.common import canonical_json_bytes
from cellstate.training.execution import (
    ContainedExecutionError,
    ContainedExecutionObservation,
    ContainedExecutionPolicy,
    ContainedTrainingObservation,
    ContainedTrainingWorkerObservation,
    ContainerCommandResult,
    ContainerCommandTimeout,
    DockerExecutor,
    ExecutionInputClosureManifest,
    RuntimeBuilderIdentity,
    RuntimeImageIdentity,
    RuntimeImageLayerIdentity,
    RuntimeImageLock,
    StagedTrainingEntry,
    StagedTrainingInventory,
    SubprocessDockerCLI,
    TrainingCodeClosureEntry,
    TrainingCodeClosureManifest,
    inventory_staged_training_tree,
    seal_staged_training_tree,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import materialize_sciplex3_k562_p1_candidate as legacy_materializer

DIGEST = "sha256:12c2faa6019fb60cdcabaa8f38f70e99be7998997b97ddb0ca59fbe2e82f1e25"
REFERENCE = f"cellstate-sciplex3-v5-runtime@{DIGEST}"
CONTAINER_ID = "b" * 64
ORPHAN_ID = "c" * 64
VOLUME_ID = "d" * 64
ORPHAN_VOLUME_ID = "e" * 64
ORPHAN_EXECUTION_ID = "orphan-run"
CODE_PAYLOAD = b"contained worker fixture\n"


def _closure() -> TrainingCodeClosureManifest:
    import hashlib

    return TrainingCodeClosureManifest(
        entries=(
            TrainingCodeClosureEntry(
                relative_path="worker.py",
                sha256=hashlib.sha256(CODE_PAYLOAD).hexdigest(),
                byte_count=len(CODE_PAYLOAD),
            ),
        )
    )


def _input_closure() -> ExecutionInputClosureManifest:
    closure = _closure()
    return ExecutionInputClosureManifest(
        training_code_closure_sha256=closure.fingerprint,
        entries=closure.entries,
    )


def _policy(**updates: object) -> ContainedExecutionPolicy:
    payload: dict[str, object] = {
        "policy_id": "sciplex3-v5-p1",
        "owner_id": "cellstate-item12-2",
        "runtime_image": RuntimeImageIdentity(reference=REFERENCE, digest=DIGEST),
        "training_code_closure_sha256": _closure().fingerprint,
        "execution_input_closure_sha256": _input_closure().fingerprint,
        "wall_clock_seconds": 3_600,
        "cleanup_timeout_seconds": 30,
        "memory_max_bytes": 4 * 1024**3,
        "memory_swap_max_bytes": 4 * 1024**3,
        "pids_limit": 64,
        "temporary_max_bytes": 256 * 1024**2,
        "snapshot_max_bytes": 3 * 1024**3,
        "observed_training_peak_memory_bytes": 1_731_055_616,
        "source_container_path": "/run/cellstate/source/source.h5ad",
        "code_container_path": "/workspace",
        "output_container_path": "/run/cellstate/output",
        "snapshot_container_path": "/run/cellstate/snapshot",
        "temporary_container_path": "/run/cellstate/tmp",
        "workdir": "/workspace",
        "environment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TMPDIR": "/run/cellstate/tmp",
        },
        "worker_command": (
            "scripts/materialize_sciplex3_k562_p1_candidate.py",
            "--worker",
        ),
    }
    payload.update(updates)
    return ContainedExecutionPolicy.model_validate(payload)


class _FakeDockerCLI:
    def __init__(
        self,
        outcome: str = "success",
        *,
        orphan: bool = False,
        image_matches: bool = True,
        cleanup_succeeds: bool = True,
        create_is_ambiguous: bool = False,
        create_times_out: bool = False,
        delayed_create_name_queries: int = 0,
        ignore_filters: bool = False,
        volume_cleanup_succeeds: bool = True,
        volume_query_succeeds: bool = True,
        container_user_matches: bool = True,
        mount_shape_matches: bool = True,
        delayed_oom_state_inspects: int = 0,
    ) -> None:
        self.outcome = outcome
        self.orphan = orphan
        self.image_matches = image_matches
        self.cleanup_succeeds = cleanup_succeeds
        self.create_is_ambiguous = create_is_ambiguous
        self.create_times_out = create_times_out
        self.delayed_create_name_queries = delayed_create_name_queries
        self.ignore_filters = ignore_filters
        self.volume_cleanup_succeeds = volume_cleanup_succeeds
        self.volume_query_succeeds = volume_query_succeeds
        self.container_user_matches = container_user_matches
        self.mount_shape_matches = mount_shape_matches
        self.delayed_oom_state_inspects = delayed_oom_state_inspects
        self.commands: list[tuple[tuple[str, ...], float | None]] = []
        self.states: dict[str, dict[str, object]] = {}
        self.labels: dict[str, dict[str, str]] = {}
        self.names: dict[str, str] = {}
        self.volumes: dict[str, str] = {}
        self.pending_create: tuple[str, dict[str, str]] | None = None
        if orphan:
            self.states[ORPHAN_ID] = {"Running": True, "OOMKilled": False, "ExitCode": 0}
            self.labels[ORPHAN_ID] = {
                "org.cellstate.contained-execution.owner": _policy().owner_id,
                "org.cellstate.contained-execution.policy-sha256": _policy().fingerprint,
                "org.cellstate.contained-execution.id": ORPHAN_EXECUTION_ID,
            }
            self.names[ORPHAN_ID] = f"cellstate-{_policy().owner_id}-{ORPHAN_EXECUTION_ID}"
            self.volumes[ORPHAN_ID] = ORPHAN_VOLUME_ID
        self.removed: set[str] = set()
        self.removed_volumes: set[str] = set()
        self.wait_timed_out = False

    def _install_created_container(self, name: str, labels: dict[str, str]) -> None:
        self.states[CONTAINER_ID] = {
            "Running": False,
            "OOMKilled": False,
            "ExitCode": 0,
        }
        self.labels[CONTAINER_ID] = labels
        self.names[CONTAINER_ID] = name
        self.volumes[CONTAINER_ID] = VOLUME_ID

    def run(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> ContainerCommandResult:
        argv = tuple(command)
        self.commands.append((argv, timeout_seconds))
        assert argv[0] == "docker"
        args = argv[1:]
        if args[:2] == ("ps", "-aq"):
            filters = [args[index + 1] for index, item in enumerate(args) if item == "--filter"]
            name_filter = next(
                (value.removeprefix("name=") for value in filters if value.startswith("name=")),
                None,
            )
            if name_filter is not None and self.pending_create is not None:
                if self.delayed_create_name_queries > 0:
                    self.delayed_create_name_queries -= 1
                else:
                    name, labels = self.pending_create
                    self._install_created_container(name, labels)
                    self.pending_create = None
            required = {
                value.removeprefix("label=").split("=", maxsplit=1)[0]: value.split("=", 2)[2]
                for value in filters
                if value.startswith("label=")
            }
            active = [
                item
                for item in self.states
                if item not in self.removed
                and (
                    self.ignore_filters
                    or all(
                        self.labels.get(item, {}).get(key) == value
                        for key, value in required.items()
                    )
                )
                and (
                    name_filter is None
                    or re.search(name_filter, f"/{self.names.get(item, '')}") is not None
                )
            ]
            return ContainerCommandResult(0, "".join(f"{item}\n" for item in active))
        if args[:2] == ("image", "inspect"):
            image = {
                "RepoDigests": [REFERENCE if self.image_matches else f"wrong@{DIGEST}"],
                "Os": "linux",
                "Architecture": "amd64",
            }
            return ContainerCommandResult(0, json.dumps(image))
        if args[0] == "create":
            labels = [args[index + 1] for index, item in enumerate(args) if item == "--label"]
            label_map = {
                item.split("=", maxsplit=1)[0]: item.split("=", maxsplit=1)[1] for item in labels
            }
            name = args[args.index("--name") + 1]
            if self.delayed_create_name_queries > 0:
                self.pending_create = (name, label_map)
            else:
                self._install_created_container(name, label_map)
            if self.create_times_out:
                raise ContainerCommandTimeout("synthetic ambiguous Docker create timeout")
            return ContainerCommandResult(
                1 if self.create_is_ambiguous else 0,
                "" if self.create_is_ambiguous else CONTAINER_ID + "\n",
            )
        if args[0] == "start":
            self.states[args[1]]["Running"] = True
            return ContainerCommandResult(0, args[1] + "\n")
        if args[0] == "wait":
            container_id = args[1]
            state = self.states[container_id]
            if (
                container_id == CONTAINER_ID
                and self.outcome == "timeout"
                and not self.wait_timed_out
            ):
                self.wait_timed_out = True
                raise ContainerCommandTimeout("synthetic wall deadline")
            if state["Running"] is True:
                state["Running"] = False
                if container_id == CONTAINER_ID and self.outcome == "oom":
                    state["ExitCode"] = 137
                elif container_id == CONTAINER_ID and self.outcome == "watchdog":
                    state["ExitCode"] = 124
                elif container_id == CONTAINER_ID and self.outcome == "ambiguous-137":
                    state["ExitCode"] = 137
                elif container_id == CONTAINER_ID and self.outcome == "failure":
                    state["ExitCode"] = 2
            return ContainerCommandResult(0, f"{state['ExitCode']}\n")
        if args[0] == "kill":
            container_id = args[-1]
            state = self.states[container_id]
            state["Running"] = False
            state["ExitCode"] = 137
            return ContainerCommandResult(0, container_id + "\n")
        if args[0] == "inspect" and "--format" in args:
            container_id = args[-1]
            template = args[args.index("--format") + 1]
            if ".Config.Labels" in template:
                return ContainerCommandResult(0, json.dumps(self.labels[container_id]))
            if ".Config.User" in template:
                configured_user = (
                    f"{os.geteuid()}:{os.getegid()}"
                    if self.container_user_matches
                    else "2147483646:2147483646"
                )
                return ContainerCommandResult(0, json.dumps(configured_user))
            if template == "{{json .Name}}":
                return ContainerCommandResult(0, json.dumps(f"/{self.names[container_id]}"))
            if ".Mounts" in template:
                policy = _policy()
                return ContainerCommandResult(
                    0,
                    json.dumps(
                        [
                            {
                                "Type": "bind",
                                "Destination": policy.source_container_path,
                                "RW": False,
                            },
                            {
                                "Type": "bind",
                                "Destination": policy.code_container_path,
                                "RW": False,
                            },
                            {
                                "Type": "bind",
                                "Destination": (
                                    policy.output_container_path
                                    if self.mount_shape_matches
                                    else "/foreign/output"
                                ),
                                "RW": True,
                            },
                            {
                                "Type": "volume",
                                "Name": self.volumes[container_id],
                                "Destination": policy.snapshot_container_path,
                                "RW": True,
                            },
                        ]
                    ),
                )
            if (
                container_id == CONTAINER_ID
                and self.outcome == "oom"
                and self.states[container_id]["ExitCode"] == 137
                and self.states[container_id]["OOMKilled"] is False
            ):
                if self.delayed_oom_state_inspects > 0:
                    self.delayed_oom_state_inspects -= 1
                else:
                    self.states[container_id]["OOMKilled"] = True
            return ContainerCommandResult(0, json.dumps(self.states[container_id]))
        if args[0] == "rm":
            container_id = args[-1]
            if not self.cleanup_succeeds:
                return ContainerCommandResult(1, stderr="synthetic removal failure")
            self.removed.add(container_id)
            if self.volume_cleanup_succeeds:
                self.removed_volumes.add(self.volumes[container_id])
            return ContainerCommandResult(0, container_id + "\n")
        if args[:2] == ("volume", "inspect"):
            return ContainerCommandResult(1 if args[2] in self.removed_volumes else 0)
        if args[:2] == ("volume", "ls"):
            if not self.volume_query_succeeds:
                return ContainerCommandResult(1, stderr="synthetic volume query failure")
            active = sorted(set(self.volumes.values()) - self.removed_volumes)
            return ContainerCommandResult(0, "".join(f"{item}\n" for item in active))
        if args[0] == "inspect":
            return ContainerCommandResult(1 if args[1] in self.removed else 0)
        raise AssertionError(argv)


def _run(
    fake: _FakeDockerCLI,
    tmp_path: Path,
    *,
    monotonic: object = time.monotonic,
    sleep: object = time.sleep,
) -> ContainedExecutionObservation:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.h5ad"
    source.write_bytes(b"source-free-fixture")
    executor = DockerExecutor(
        _policy(),
        cli=fake,
        monotonic=monotonic,
        sleep=sleep,
        lock_root=tmp_path / "locks",
        staging_root=tmp_path / "staging",
        canonical_publication_root=tmp_path / "canonical-publication",
        execution_input_closure=_input_closure(),
    )
    code = tmp_path / "code"
    code.mkdir()
    (code / "worker.py").write_bytes(CODE_PAYLOAD)
    return executor.run(
        execution_id="run-0001",
        source_path=source,
        code_path=code,
        output_path=executor.output_stage_path("run-0001"),
    )


class _AdvancingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def test_policy_requires_canonical_digest_limits_isolation_and_bytes() -> None:
    with pytest.raises(ValidationError, match="digest"):
        RuntimeImageIdentity(
            reference="ghcr.io/example/cellstate-runtime:latest",
            digest=DIGEST,
        )
    with pytest.raises(ValidationError, match="swap"):
        _policy(memory_swap_max_bytes=8 * 1024**3)
    with pytest.raises(ValidationError):
        _policy(wall_clock_seconds="3600")
    with pytest.raises(ValidationError, match="sorted"):
        _policy(environment={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    with pytest.raises(ValidationError, match="overlap"):
        _policy(output_container_path="/run/cellstate/source/source.h5ad/output")
    with pytest.raises(ValidationError, match="workdir"):
        _policy(workdir="/run/cellstate/output/work")

    policy = _policy()
    payload = canonical_json_bytes(policy.model_dump(mode="json"))
    assert ContainedExecutionPolicy.from_canonical_json(payload) == policy
    with pytest.raises(ContainedExecutionError, match="canonical JSON"):
        ContainedExecutionPolicy.from_canonical_json(payload + b"\n")
    assert policy.fingerprint == _policy().fingerprint
    protected_source_bytes = 2_526_631_614
    safety_margin_bytes = 64 * 1024**2
    assert protected_source_bytes < policy.snapshot_max_bytes
    assert policy.observed_training_peak_memory_bytes < policy.memory_max_bytes
    assert policy.memory_max_bytes - policy.observed_training_peak_memory_bytes < (
        protected_source_bytes + safety_margin_bytes
    )

    with pytest.raises(ValidationError, match="contradicts"):
        ContainedExecutionObservation(
            execution_id="run-0001",
            policy_fingerprint=policy.fingerprint,
            runtime_image_digest=DIGEST,
            container_user_mode="host-effective-uid-gid",
            observed_container_uid=os.geteuid(),
            observed_container_gid=os.getegid(),
            outcome="success",
            exit_code=137,
            timed_out=True,
            oom_killed=False,
            parent_wall_clock_elapsed_seconds=3_600.0,
        )
    with pytest.raises(ValidationError, match="contradicts"):
        ContainedExecutionObservation(
            execution_id="run-0001",
            policy_fingerprint=policy.fingerprint,
            runtime_image_digest=DIGEST,
            container_user_mode="host-effective-uid-gid",
            observed_container_uid=os.geteuid(),
            observed_container_gid=os.getegid(),
            outcome="timeout",
            exit_code=137,
            timed_out=False,
            worker_watchdog_timed_out=True,
            oom_killed=False,
            parent_wall_clock_elapsed_seconds=20.0,
        )


def test_execution_observation_v1_2_round_trip_rejects_ambiguous_v1_1_watchdog() -> None:
    policy = _policy()
    observation = ContainedExecutionObservation(
        execution_id="run-0001",
        policy_fingerprint=policy.fingerprint,
        runtime_image_digest=DIGEST,
        container_user_mode="host-effective-uid-gid",
        observed_container_uid=os.geteuid(),
        observed_container_gid=os.getegid(),
        outcome="timeout",
        exit_code=124,
        timed_out=False,
        worker_watchdog_timed_out=True,
        oom_killed=False,
        parent_wall_clock_elapsed_seconds=20.0,
    )
    payload = canonical_json_bytes(observation.model_dump(mode="json"))
    assert observation.artifact_schema_version == "1.2.0"
    assert ContainedExecutionObservation.model_validate_json(payload) == observation

    old_payload = observation.model_dump(mode="json")
    old_payload["artifact_schema_version"] = "1.1.0"
    old_payload["exit_code"] = 137
    with pytest.raises(ValidationError):
        ContainedExecutionObservation.model_validate(old_payload)


def test_subprocess_cli_translates_success_timeout_and_unavailable_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = SubprocessDockerCLI()
    completed = subprocess.CompletedProcess(
        args=("docker", "version"),
        returncode=3,
        stdout="stdout",
        stderr="stderr",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)
    assert cli.run(("docker", "version"), timeout_seconds=1.0) == ContainerCommandResult(
        3, "stdout", "stderr"
    )

    def timed_out(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise subprocess.TimeoutExpired(("docker", "wait"), 1.0)

    monkeypatch.setattr(subprocess, "run", timed_out)
    with pytest.raises(ContainerCommandTimeout, match="parent deadline"):
        cli.run(("docker", "wait"), timeout_seconds=1.0)

    def unavailable(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise FileNotFoundError("docker")

    monkeypatch.setattr(subprocess, "run", unavailable)
    with pytest.raises(ContainedExecutionError, match="unavailable"):
        cli.run(("docker", "version"), timeout_seconds=1.0)
    with pytest.raises(TypeError, match="return code"):
        ContainerCommandResult("0")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="output"):
        ContainerCommandResult(0, stdout=b"bytes")  # type: ignore[arg-type]


def test_contract_models_reject_ambiguous_paths_limits_and_typed_evidence() -> None:
    with pytest.raises(ValidationError, match="pinned"):
        RuntimeImageIdentity(
            reference="example.invalid/runtime@sha256:" + "a" * 64,
            digest=DIGEST,
        )
    with pytest.raises(ValidationError, match="whitespace"):
        RuntimeImageIdentity(
            reference=f"example.invalid/runtime @{DIGEST}",
            digest=DIGEST,
        )
    image_lock = RuntimeImageLock(
        runtime_image=RuntimeImageIdentity(reference=REFERENCE, digest=DIGEST),
        builder=RuntimeBuilderIdentity(
            buildx_version="v0.28.0",
            buildx_commit="1" * 40,
            buildkit_version="v0.24.0",
            buildkit_image_digest="sha256:" + "2" * 64,
            dockerfile_frontend_digest="sha256:" + "3" * 64,
            dockerfile_sha256="4" * 64,
            requirements_sha256="5" * 64,
            source_date_epoch=1_786_406_400,
            no_cache=True,
            provenance_attestation_disabled=True,
            image_tag="cellstate-sciplex3-v5-runtime:20260811-locked",
            output_options=("type=oci",),
        ),
        archive_sha256="6" * 64,
        oci_index_digest="sha256:" + "7" * 64,
        config_digest="sha256:" + "8" * 64,
        layers=(
            RuntimeImageLayerIdentity(
                digest="sha256:" + "9" * 64,
                byte_count=1,
            ),
        ),
        training_code_closure_sha256=_closure().fingerprint,
        image_provenance_sha256="a" * 64,
    )
    assert len(image_lock.fingerprint) == 64

    first = TrainingCodeClosureEntry(relative_path="a.py", sha256="a" * 64, byte_count=1)
    second = TrainingCodeClosureEntry(relative_path="z.py", sha256="b" * 64, byte_count=1)
    with pytest.raises(ValidationError, match="unique and sorted"):
        TrainingCodeClosureManifest(entries=(second, first))
    with pytest.raises(ValidationError, match="unique and sorted"):
        ExecutionInputClosureManifest(
            training_code_closure_sha256="c" * 64,
            entries=(second, first),
        )
    with pytest.raises(ValidationError, match="observation metadata"):
        StagedTrainingEntry(
            relative_path="contained-worker-observation.json",
            artifact_role="support",
            sha256="d" * 64,
            byte_count=1,
        )
    with pytest.raises(ValidationError, match="artifact role"):
        StagedTrainingEntry(
            relative_path="candidate-model.json",
            artifact_role="support",
            sha256="d" * 64,
            byte_count=1,
        )
    entry = StagedTrainingEntry(
        relative_path="candidate-model.json",
        artifact_role="model_artifact",
        sha256="d" * 64,
        byte_count=1,
    )
    with pytest.raises(ValidationError, match="exclusions"):
        StagedTrainingInventory(
            excluded_observation_paths=("one", "two"),
            entries=(entry,),
        )
    support = StagedTrainingEntry(
        relative_path="z-support.json",
        artifact_role="support",
        sha256="e" * 64,
        byte_count=1,
    )
    with pytest.raises(ValidationError, match="unique and sorted"):
        StagedTrainingInventory(entries=(support, entry))

    invalid_policies = (
        ({"policy_id": "Not Canonical"}, "identifiers"),
        ({"source_container_path": "relative/source"}, "absolute POSIX"),
        ({"worker_command": ("",)}, "argument vector"),
        ({"temporary_max_bytes": 5 * 1024**3}, "temporary storage"),
        ({"observed_training_peak_memory_bytes": 4 * 1024**3}, "positive cgroup"),
        (
            {
                "environment": {
                    "TMPDIR": "/run/cellstate/tmp",
                    "lowercase": "invalid",
                }
            },
            "environment",
        ),
        ({"environment": {"OMP_NUM_THREADS": "1"}}, "TMPDIR"),
    )
    for update, message in invalid_policies:
        with pytest.raises(ValidationError, match=message):
            _policy(**update)
    with pytest.raises(ContainedExecutionError, match="policy is invalid"):
        ContainedExecutionPolicy.from_canonical_json(b"{}")

    inventory = StagedTrainingInventory(entries=(entry,))
    worker_fields: dict[str, object] = {
        "execution_id": "typed-fit",
        "training_plan_fingerprint": "1" * 64,
        "policy_fingerprint": "2" * 64,
        "runtime_image_digest": DIGEST,
        "training_code_closure_sha256": "3" * 64,
        "execution_input_closure_sha256": "4" * 64,
        "expected_source_sha256": "5" * 64,
        "source_pre_sha256": "5" * 64,
        "source_post_sha256": "5" * 64,
        "expected_source_byte_count": 10,
        "source_pre_byte_count": 10,
        "source_post_byte_count": 10,
        "staged_inventory": inventory,
        "staged_tree_sha256": inventory.fingerprint,
        "staged_file_count": 1,
    }
    with pytest.raises(ValidationError, match="reauthenticate"):
        ContainedTrainingWorkerObservation.model_validate(
            {**worker_fields, "source_post_sha256": "6" * 64}
        )
    with pytest.raises(ValidationError, match="staged-tree summary"):
        ContainedTrainingWorkerObservation.model_validate(
            {**worker_fields, "staged_tree_sha256": "7" * 64}
        )
    worker = ContainedTrainingWorkerObservation.model_validate(worker_fields)
    assert len(worker.fingerprint) == 64
    success = ContainedExecutionObservation(
        execution_id="typed-fit",
        policy_fingerprint="2" * 64,
        runtime_image_digest=DIGEST,
        container_user_mode="host-effective-uid-gid",
        observed_container_uid=1000,
        observed_container_gid=1000,
        outcome="success",
        exit_code=0,
        timed_out=False,
        oom_killed=False,
        parent_wall_clock_elapsed_seconds=10.0,
    )
    training_fields: dict[str, object] = {
        "training_plan_fingerprint": "1" * 64,
        "policy_fingerprint": "2" * 64,
        "runtime_image_digest": DIGEST,
        "training_code_closure_sha256": "3" * 64,
        "execution_input_closure_sha256": "4" * 64,
        "staged_inventory": inventory,
        "staged_tree_sha256": inventory.fingerprint,
        "worker_observation": worker,
        "execution_observation": success,
        "wall_clock_limit_seconds": 3_600,
        "memory_max_bytes": 4 * 1024**3,
        "memory_swap_max_bytes": 4 * 1024**3,
    }
    training = ContainedTrainingObservation.model_validate(training_fields)
    assert len(training.fingerprint) == 64
    timeout = success.model_copy(update={"outcome": "timeout", "exit_code": 137, "timed_out": True})
    with pytest.raises(ValidationError, match="successful worker tree"):
        ContainedTrainingObservation.model_validate(
            {**training_fields, "execution_observation": timeout}
        )


def test_stage_inventory_and_seal_fail_closed_on_missing_roots_links_and_drift(
    tmp_path: Path,
) -> None:
    with pytest.raises(ContainedExecutionError, match="inspect staged training root"):
        inventory_staged_training_tree(tmp_path / "missing")
    root_file = tmp_path / "root-file"
    root_file.write_bytes(b"not-a-directory")
    with pytest.raises(ContainedExecutionError, match="real directory"):
        inventory_staged_training_tree(root_file)

    linked_tree = tmp_path / "linked-tree"
    linked_tree.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (linked_tree / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ContainedExecutionError, match="directory link"):
        inventory_staged_training_tree(linked_tree)

    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "candidate-model.json").write_bytes(b"model")
    inventory = inventory_staged_training_tree(stage)
    substituted = StagedTrainingInventory(
        entries=(
            StagedTrainingEntry(
                relative_path="candidate-model.json",
                artifact_role="model_artifact",
                sha256="f" * 64,
                byte_count=5,
            ),
        )
    )
    with pytest.raises(ContainedExecutionError, match="changed before sealing"):
        seal_staged_training_tree(stage, expected_inventory=substituted)
    seal_staged_training_tree(stage, expected_inventory=inventory)


def test_create_command_freezes_aggregate_cgroup_and_process_tree_isolation(tmp_path: Path) -> None:
    policy = _policy()
    executor = DockerExecutor(policy, cli=_FakeDockerCLI())
    source = (tmp_path / "source.h5ad").absolute()
    output = (tmp_path / "output").absolute()
    code = (tmp_path / "code").absolute()
    command = executor.build_create_command(
        execution_id="run-0001", source_path=source, code_path=code, output_path=output
    )

    assert command[:2] == ("docker", "create")
    assert command[command.index("--platform") : command.index("--platform") + 2] == (
        "--platform",
        "linux/amd64",
    )
    for flag, value in (
        ("--pull", "never"),
        ("--memory", str(4 * 1024**3)),
        ("--memory-swap", str(4 * 1024**3)),
        ("--pids-limit", "64"),
        ("--user", f"{os.geteuid()}:{os.getegid()}"),
        ("--network", "none"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
    ):
        index = command.index(flag)
        assert command[index + 1] == value
    assert "--read-only" in command
    assert "--init" in command
    tmpfs_index = command.index("--tmpfs")
    assert command[tmpfs_index + 1] == (
        "/run/cellstate/tmp:rw,noexec,nosuid,nodev,size=268435456,mode=0700,"
        f"uid={os.geteuid()},gid={os.getegid()}"
    )
    assert REFERENCE in command
    mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
    assert mounts == [
        f"type=bind,source={source},target={policy.source_container_path},readonly",
        f"type=bind,source={code},target={policy.code_container_path},readonly",
        f"type=bind,source={output},target={policy.output_container_path}",
        f"type=volume,target={policy.snapshot_container_path}",
    ]
    assert tuple(command[index + 1] for index, value in enumerate(command) if value == "--env") == (
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "TMPDIR=/run/cellstate/tmp",
    )


def test_success_preflights_digest_recovers_orphans_and_removes_complete_tree(
    tmp_path: Path,
) -> None:
    fake = _FakeDockerCLI(orphan=True)
    observation = _run(fake, tmp_path)
    assert observation.outcome == "success"
    assert observation.exit_code == 0
    assert observation.container_user_mode == "host-effective-uid-gid"
    assert observation.observed_container_uid == os.geteuid()
    assert observation.observed_container_gid == os.getegid()
    assert observation.container_removed
    assert observation.snapshot_volume_removed
    assert observation.process_tree_cleaned
    assert not observation.canonical_publication_performed
    assert fake.removed == {ORPHAN_ID, CONTAINER_ID}
    assert fake.removed_volumes == {ORPHAN_VOLUME_ID, VOLUME_ID}

    verbs = [command[1] for command, _ in fake.commands]
    assert verbs.index("ps") < verbs.index("image") < verbs.index("create") < verbs.index("start")
    assert (tmp_path / "source.h5ad").read_bytes() == b"source-free-fixture"
    create = next(command for command, _ in fake.commands if command[1] == "create")
    code_mount = next(
        create[index + 1]
        for index, item in enumerate(create)
        if item == "--mount" and "target=/workspace" in create[index + 1]
    )
    assert "/run-0001/code,target=/workspace,readonly" in code_mount
    staged_code = tmp_path / "staging/cellstate-item12-2/run-0001/code"
    assert staged_code.stat().st_mode & 0o777 == 0o555
    assert (staged_code / "worker.py").stat().st_mode & 0o777 == 0o400


def test_container_user_readback_must_match_host_effective_policy(tmp_path: Path) -> None:
    fake = _FakeDockerCLI(container_user_matches=False)
    with pytest.raises(ContainedExecutionError, match="user differs"):
        _run(fake, tmp_path)
    assert fake.removed == {CONTAINER_ID}
    assert fake.removed_volumes == {VOLUME_ID}


def test_ambiguous_create_recovers_its_exact_labeled_container(
    tmp_path: Path,
) -> None:
    clock = _AdvancingClock()
    ambiguous = _FakeDockerCLI(create_is_ambiguous=True)
    with pytest.raises(ContainedExecutionError, match="failed to create"):
        _run(
            ambiguous,
            tmp_path / "ambiguous",
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    assert CONTAINER_ID in ambiguous.removed


def test_new_policy_recovers_authenticated_historical_policy_orphan_and_volume(
    tmp_path: Path,
) -> None:
    historical = _FakeDockerCLI(orphan=True)
    historical.labels[ORPHAN_ID]["org.cellstate.contained-execution.policy-sha256"] = "d" * 64
    new_policy = _policy(worker_command=("new-worker.py",))
    assert new_policy.fingerprint != "d" * 64
    recovered = DockerExecutor(
        new_policy, cli=historical, lock_root=tmp_path / "locks"
    ).recover_owned_containers()
    assert recovered == (ORPHAN_ID,)
    assert historical.removed == {ORPHAN_ID}
    assert historical.removed_volumes == {ORPHAN_VOLUME_ID}


@pytest.mark.parametrize("malformation", ("name", "mount"))
def test_same_owner_malformed_historical_resource_fails_closed_without_removal(
    tmp_path: Path,
    malformation: str,
) -> None:
    malformed = _FakeDockerCLI(
        orphan=True,
        mount_shape_matches=malformation != "mount",
    )
    malformed.labels[ORPHAN_ID]["org.cellstate.contained-execution.policy-sha256"] = "d" * 64
    if malformation == "name":
        malformed.names[ORPHAN_ID] = "unrelated-container"
    executor = DockerExecutor(_policy(), cli=malformed, lock_root=tmp_path / malformation)
    with pytest.raises(ContainedExecutionError, match=r"namespace|mount shape"):
        executor.recover_owned_containers()
    assert ORPHAN_ID not in malformed.removed
    assert ORPHAN_VOLUME_ID not in malformed.removed_volumes


def test_create_timeout_polls_exact_name_until_delayed_container_and_volume_are_absent(
    tmp_path: Path,
) -> None:
    clock = _AdvancingClock()
    delayed = _FakeDockerCLI(
        create_times_out=True,
        delayed_create_name_queries=3,
    )
    with pytest.raises(ContainedExecutionError, match="crossed the wall deadline"):
        _run(
            delayed,
            tmp_path / "delayed-create",
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    assert CONTAINER_ID in delayed.removed
    assert VOLUME_ID in delayed.removed_volumes
    exact_name_queries = [
        command
        for command, _ in delayed.commands
        if command[1:3] == ("ps", "-aq")
        and any(item.startswith("name=^/cellstate") for item in command)
    ]
    assert len(exact_name_queries) > 3
    assert clock.value >= _policy().cleanup_timeout_seconds


def test_recovery_removes_created_but_never_started_container_without_waiting(
    tmp_path: Path,
) -> None:
    fake = _FakeDockerCLI(orphan=True)
    fake.states[ORPHAN_ID]["Running"] = False
    recovered = DockerExecutor(
        _policy(), cli=fake, lock_root=tmp_path / "locks"
    ).recover_owned_containers()
    assert recovered == (ORPHAN_ID,)
    assert ORPHAN_ID in fake.removed
    assert all(command != ("docker", "wait", ORPHAN_ID) for command, _ in fake.commands)


def test_parent_deadline_starts_before_container_create_and_covers_start_and_wait(
    tmp_path: Path,
) -> None:
    moments = iter((100.0, 100.0, 100.0, 100.0, 100.0, 101.0, 102.0, 103.0, 104.0))
    fake = _FakeDockerCLI()
    source = tmp_path / "source.h5ad"
    source.write_bytes(b"deadline-fixture")
    executor = DockerExecutor(
        _policy(),
        cli=fake,
        monotonic=moments.__next__,
        lock_root=tmp_path / "locks",
        staging_root=tmp_path / "staging",
        canonical_publication_root=tmp_path / "canonical-publication",
        execution_input_closure=_input_closure(),
    )
    code = tmp_path / "code"
    code.mkdir()
    (code / "worker.py").write_bytes(CODE_PAYLOAD)
    observation = executor.run(
        execution_id="run-deadline",
        source_path=source,
        code_path=code,
        output_path=executor.output_stage_path("run-deadline"),
    )
    assert observation.outcome == "success"
    timed = {
        command[1]: timeout
        for command, timeout in fake.commands
        if command[1] in {"create", "start", "wait"} and command[-1] == CONTAINER_ID
    }
    create_timeout = next(timeout for command, timeout in fake.commands if command[1] == "create")
    assert create_timeout == 3_600.0
    assert timed["start"] == 3_599.0
    assert timed["wait"] == 3_598.0
    assert observation.parent_wall_clock_elapsed_seconds == 3.0
    assert (tmp_path / "locks/cellstate-item12-2.lock").is_file()


def test_parent_timeout_kills_waits_and_removes_descendant_container_tree(tmp_path: Path) -> None:
    fake = _FakeDockerCLI("timeout")
    observation = _run(fake, tmp_path)
    assert observation.outcome == "timeout"
    assert observation.timed_out
    assert observation.exit_code == 137
    assert CONTAINER_ID in fake.removed
    commands = [command for command, _ in fake.commands]
    assert ("docker", "kill", "--signal", "KILL", CONTAINER_ID) in commands
    kill_index = commands.index(("docker", "kill", "--signal", "KILL", CONTAINER_ID))
    remove_index = next(index for index, command in enumerate(commands) if command[1] == "rm")
    assert kill_index < remove_index
    assert not (tmp_path / "staging/cellstate-item12-2/run-0001/current.json").exists()


def test_worker_watchdog_exit_is_typed_as_timeout_without_inventing_parent_timeout(
    tmp_path: Path,
) -> None:
    observation = _run(_FakeDockerCLI("watchdog"), tmp_path)
    assert observation.outcome == "timeout"
    assert observation.exit_code == 124
    assert not observation.timed_out
    assert observation.worker_watchdog_timed_out
    assert not observation.oom_killed


def test_cgroup_oom_and_worker_failure_never_publish_and_are_distinguished(tmp_path: Path) -> None:
    oom = _run(_FakeDockerCLI("oom"), tmp_path / "oom")
    failed = _run(_FakeDockerCLI("failure"), tmp_path / "failure")
    assert (oom.outcome, oom.exit_code, oom.oom_killed) == ("oom_killed", 137, True)
    assert (failed.outcome, failed.exit_code, failed.oom_killed) == (
        "worker_failure",
        2,
        False,
    )
    assert not (tmp_path / "oom/stage/current.json").exists()
    assert not (tmp_path / "failure/stage/current.json").exists()


def test_late_positive_oom_state_is_boundedly_stabilized_before_removal(tmp_path: Path) -> None:
    clock = _AdvancingClock()
    fake = _FakeDockerCLI("oom", delayed_oom_state_inspects=2)
    observation = _run(
        fake,
        tmp_path,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert (observation.outcome, observation.exit_code, observation.oom_killed) == (
        "oom_killed",
        137,
        True,
    )
    assert not observation.worker_watchdog_timed_out
    assert clock.value == pytest.approx(0.2)


def test_unresolved_exit_137_is_fail_closed_not_invented_as_watchdog(tmp_path: Path) -> None:
    clock = _AdvancingClock()
    observation = _run(
        _FakeDockerCLI("ambiguous-137"),
        tmp_path,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert (observation.outcome, observation.exit_code, observation.oom_killed) == (
        "worker_failure",
        137,
        False,
    )
    assert not observation.timed_out
    assert not observation.worker_watchdog_timed_out
    assert clock.value == pytest.approx(0.5)


def test_execution_id_replay_preserves_prior_terminal_evidence_without_docker(
    tmp_path: Path,
) -> None:
    _run(_FakeDockerCLI(), tmp_path)
    execution_root = tmp_path / "staging/cellstate-item12-2/run-0001"
    terminal = execution_root / "contained-training-terminal-observation.json"
    terminal.write_bytes(b"prior-terminal-evidence")
    source = tmp_path / "source.h5ad"
    code = tmp_path / "code"
    replay_cli = _FakeDockerCLI()
    replay = DockerExecutor(
        _policy(),
        cli=replay_cli,
        lock_root=tmp_path / "locks",
        staging_root=tmp_path / "staging",
        canonical_publication_root=tmp_path / "canonical-publication",
        execution_input_closure=_input_closure(),
    )
    with pytest.raises(ContainedExecutionError, match="already been consumed"):
        replay.run(
            execution_id="run-0001",
            source_path=source,
            code_path=code,
            output_path=replay.output_stage_path("run-0001"),
        )
    assert terminal.read_bytes() == b"prior-terminal-evidence"
    assert replay_cli.commands == []


def test_output_stage_is_preclaimed_before_owner_lock_failure_for_terminal_quarantine(
    tmp_path: Path,
) -> None:
    lock_root = tmp_path / "lock-root"
    lock_root.write_bytes(b"synthetic-lock-root-failure")
    executor = DockerExecutor(
        _policy(),
        cli=_FakeDockerCLI(),
        lock_root=lock_root,
        staging_root=tmp_path / "staging",
        canonical_publication_root=tmp_path / "canonical-publication",
        execution_input_closure=_input_closure(),
    )
    output = executor.output_stage_path("lock-failure")
    with pytest.raises(ContainedExecutionError, match="stable execution-owner lock"):
        executor.run(
            execution_id="lock-failure",
            source_path=tmp_path / "source-that-must-not-be-inspected.h5ad",
            code_path=tmp_path / "code-that-must-not-be-inspected",
            output_path=output,
        )
    assert output.is_dir()
    assert tuple(output.iterdir()) == ()
    assert (output.parent / ".cellstate-execution-owner.json").is_file()


def test_output_stage_is_executor_owned_and_cannot_alias_canonical_publication(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.h5ad"
    source.write_bytes(b"fixture")
    canonical = tmp_path / "canonical"
    executor = DockerExecutor(
        _policy(),
        cli=_FakeDockerCLI(),
        lock_root=tmp_path / "locks",
        staging_root=canonical / "nested-stage",
        canonical_publication_root=canonical,
        execution_input_closure=_input_closure(),
    )
    code = tmp_path / "code"
    code.mkdir()
    (code / "worker.py").write_bytes(CODE_PAYLOAD)
    with pytest.raises(ContainedExecutionError, match="overlap"):
        executor.run(
            execution_id="run-alias",
            source_path=source,
            code_path=code,
            output_path=executor.output_stage_path("run-alias"),
        )


def test_host_symlink_ancestors_and_canonical_source_hardlinks_are_rejected(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(canonical, target_is_directory=True)
    source = tmp_path / "source.h5ad"
    source.write_bytes(b"fixture")
    code = tmp_path / "code"
    code.mkdir()
    (code / "worker.py").write_bytes(CODE_PAYLOAD)
    symlinked = DockerExecutor(
        _policy(),
        cli=_FakeDockerCLI(),
        lock_root=tmp_path / "locks",
        staging_root=alias / "stage",
        canonical_publication_root=canonical,
        execution_input_closure=_input_closure(),
    )
    with pytest.raises(ContainedExecutionError, match="symlinked ancestor"):
        symlinked.run(
            execution_id="run-symlink",
            source_path=source,
            code_path=code,
            output_path=symlinked.output_stage_path("run-symlink"),
        )
    assert not tuple(canonical.iterdir())

    canonical_source = canonical / "source.h5ad"
    canonical_source.write_bytes(b"protected")
    alias_source = tmp_path / "hardlinked-source.h5ad"
    alias_source.hardlink_to(canonical_source)
    hardlinked = DockerExecutor(
        _policy(),
        cli=_FakeDockerCLI(),
        lock_root=tmp_path / "hardlink-locks",
        staging_root=tmp_path / "hardlink-stage",
        canonical_publication_root=canonical,
        execution_input_closure=_input_closure(),
    )
    with pytest.raises(ContainedExecutionError, match="inode closure"):
        hardlinked.run(
            execution_id="run-hardlink",
            source_path=alias_source,
            code_path=code,
            output_path=hardlinked.output_stage_path("run-hardlink"),
        )


def test_image_mismatch_and_unprovable_cleanup_fail_closed_before_worker(tmp_path: Path) -> None:
    image_mismatch = _FakeDockerCLI(image_matches=False)
    with pytest.raises(ContainedExecutionError, match="frozen digest"):
        _run(image_mismatch, tmp_path / "image")
    assert all(command[1] != "create" for command, _ in image_mismatch.commands)

    cleanup_failure = _FakeDockerCLI(cleanup_succeeds=False)
    with pytest.raises(ContainedExecutionError, match="cleanup could not be proved"):
        _run(cleanup_failure, tmp_path / "cleanup")

    volume_remained = _FakeDockerCLI(volume_cleanup_succeeds=False)
    with pytest.raises(ContainedExecutionError, match="cleanup could not be proved"):
        _run(volume_remained, tmp_path / "volume-remained")
    assert VOLUME_ID not in volume_remained.removed_volumes

    volume_query_failed = _FakeDockerCLI(volume_query_succeeds=False)
    with pytest.raises(ContainedExecutionError, match="cleanup could not be proved"):
        _run(volume_query_failed, tmp_path / "volume-query")


def test_parent_reinventories_and_seals_exact_worker_stage(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    (stage / "nested").mkdir(parents=True)
    (stage / "candidate-model.json").write_bytes(b"model")
    (stage / "nested/plan.json").write_bytes(b"plan")
    (stage / "contained-worker-observation.json").write_bytes(b"metadata")
    inventory = inventory_staged_training_tree(stage)
    assert isinstance(inventory, StagedTrainingInventory)
    assert tuple(entry.relative_path for entry in inventory.entries) == (
        "candidate-model.json",
        "nested/plan.json",
    )
    (stage / "contained-worker-observation.json").write_bytes(b"new metadata")
    assert inventory_staged_training_tree(stage) == inventory
    (stage / "candidate-model.json").write_bytes(b"substituted")
    assert inventory_staged_training_tree(stage) != inventory
    (stage / "candidate-model.json").write_bytes(b"model")
    seal_staged_training_tree(stage, expected_inventory=inventory)
    assert stage.stat().st_mode & 0o777 == 0o555
    assert all(path.stat().st_mode & 0o777 == 0o444 for path in stage.rglob("*") if path.is_file())


def test_legacy_direct_materializer_retires_before_source_or_publication_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_path_access(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("retired direct materializer touched a filesystem path")

    monkeypatch.setattr(Path, "open", forbidden_path_access)
    monkeypatch.setattr(Path, "read_bytes", forbidden_path_access)
    with pytest.raises(
        legacy_materializer.CandidateMaterializationError,
        match="legacy direct materialization is retired",
    ):
        legacy_materializer.materialize(
            tmp_path / "protected-source.h5ad",
            tmp_path / "canonical-publication",
            repository_root=tmp_path,
        )
    assert not (tmp_path / "canonical-publication").exists()
