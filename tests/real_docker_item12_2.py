"""Explicit-only real-Docker probes for Item 12.2 containment.

This file intentionally does not match the repository's default ``test_*.py`` pattern. CI invokes
it directly after reconstructing and loading the exact locked OCI image; missing Docker or a
missing image is therefore a failure, never a skip.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cellstate.domain.common import canonical_json_bytes
from cellstate.training.execution import (
    ContainedExecutionObservation,
    ContainedExecutionPolicy,
    DockerExecutor,
    ExecutionInputClosureManifest,
    RuntimeImageIdentity,
    TrainingCodeClosureEntry,
    TrainingCodeClosureManifest,
    inventory_staged_training_tree,
    seal_staged_training_tree,
)

IMAGE_DIGEST = "sha256:12c2faa6019fb60cdcabaa8f38f70e99be7998997b97ddb0ca59fbe2e82f1e25"
IMAGE_REFERENCE = f"cellstate-sciplex3-v5-runtime@{IMAGE_DIGEST}"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROBE = b"""\
import os
import subprocess
import sys
import time
from pathlib import Path

mode = sys.argv[1]
output = Path(sys.argv[2])
if mode == "success":
    nested = output / "nested"
    nested.mkdir(mode=0o700)
    result = nested / "result.txt"
    result.write_text("contained\\n", encoding="utf-8")
    result.chmod(0o400)
elif mode == "sleep-tree":
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    time.sleep(120)
elif mode == "oom":
    blocks = []
    while True:
        block = bytearray(8 * 1024 * 1024)
        for offset in range(0, len(block), 4096):
            block[offset] = 1
        blocks.append(block)
else:
    raise SystemExit(64)
"""


def _contracts(
    *, owner_id: str, mode: str, wall_seconds: int
) -> tuple[
    ContainedExecutionPolicy,
    ExecutionInputClosureManifest,
]:
    entry = TrainingCodeClosureEntry(
        relative_path="probe.py",
        sha256=hashlib.sha256(PROBE).hexdigest(),
        byte_count=len(PROBE),
    )
    code = TrainingCodeClosureManifest(entries=(entry,))
    inputs = ExecutionInputClosureManifest(
        training_code_closure_sha256=code.fingerprint,
        entries=(entry,),
    )
    watchdog_seconds = 3 if mode == "sleep-tree" and wall_seconds > 3 else 20
    policy = ContainedExecutionPolicy(
        policy_id=f"{owner_id}-policy",
        owner_id=owner_id,
        runtime_image=RuntimeImageIdentity(
            reference=IMAGE_REFERENCE,
            digest=IMAGE_DIGEST,
        ),
        training_code_closure_sha256=code.fingerprint,
        execution_input_closure_sha256=inputs.fingerprint,
        wall_clock_seconds=wall_seconds,
        cleanup_timeout_seconds=15,
        memory_max_bytes=64 * 1024**2 if mode == "oom" else 256 * 1024**2,
        memory_swap_max_bytes=64 * 1024**2 if mode == "oom" else 256 * 1024**2,
        pids_limit=32,
        temporary_max_bytes=4 * 1024**2,
        snapshot_max_bytes=1024**2,
        observed_training_peak_memory_bytes=1,
        source_container_path="/run/cellstate/source/source.h5ad",
        code_container_path="/workspace",
        output_container_path="/run/cellstate/output",
        snapshot_container_path="/run/cellstate/snapshot",
        temporary_container_path="/run/cellstate/tmp",
        workdir="/workspace",
        environment={"PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": "/run/cellstate/tmp"},
        worker_command=(
            "--signal=TERM",
            "--kill-after=1s",
            str(watchdog_seconds),
            "/opt/runtime/bin/python",
            "/workspace/probe.py",
            mode,
            "/run/cellstate/output",
        ),
    )
    return policy, inputs


def _executor(
    tmp_path: Path,
    *,
    owner_id: str,
    mode: str,
    wall_seconds: int,
) -> tuple[DockerExecutor, Path, Path, ContainedExecutionPolicy, ExecutionInputClosureManifest]:
    policy, inputs = _contracts(owner_id=owner_id, mode=mode, wall_seconds=wall_seconds)
    code = tmp_path / "code"
    code.mkdir()
    (code / "probe.py").write_bytes(PROBE)
    source = tmp_path / "source.h5ad"
    source.write_bytes(b"source-free-real-docker-probe\n")
    executor = DockerExecutor(
        policy,
        lock_root=tmp_path / "locks",
        staging_root=tmp_path / "stage",
        canonical_publication_root=tmp_path / "canonical",
        execution_input_closure=inputs,
    )
    return executor, source, code, policy, inputs


@pytest.fixture(scope="module", autouse=True)
def _exact_image_is_loaded() -> None:
    completed = subprocess.run(
        ["docker", "image", "inspect", IMAGE_REFERENCE],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("mode", "wall_seconds", "expected"),
    (
        ("success", 10, ("success", 0, False)),
        ("sleep-tree", 1, ("timeout", 137, False)),
    ),
)
def test_real_docker_outcomes_remove_tree_and_anonymous_volume(
    tmp_path: Path,
    mode: str,
    wall_seconds: int,
    expected: tuple[str, int, bool],
) -> None:
    executor, source, code, _, _ = _executor(
        tmp_path,
        owner_id=f"item12-real-{mode}",
        mode=mode,
        wall_seconds=wall_seconds,
    )
    observation = executor.run(
        execution_id="probe",
        source_path=source,
        code_path=code,
        output_path=executor.output_stage_path("probe"),
    )
    assert isinstance(observation, ContainedExecutionObservation)
    assert (observation.outcome, observation.exit_code, observation.oom_killed) == expected
    assert observation.container_removed
    assert observation.snapshot_volume_removed
    assert observation.process_tree_cleaned
    assert not observation.canonical_publication_performed
    if mode == "success":
        result_path = executor.output_stage_path("probe") / "nested/result.txt"
        assert result_path.read_text() == "contained\n"
        result_state = result_path.lstat()
        result_mode = stat.S_IMODE(result_state.st_mode)
        assert stat.S_ISREG(result_state.st_mode)
        assert (result_state.st_uid, result_state.st_gid) == (os.geteuid(), os.getegid())
        assert os.access(result_path, os.R_OK)
        assert result_mode & 0o077 == 0
        if sys.platform == "linux":
            assert result_mode == 0o400
        inventory = inventory_staged_training_tree(executor.output_stage_path("probe"))
        assert tuple(entry.relative_path for entry in inventory.entries) == ("nested/result.txt",)
        seal_staged_training_tree(
            executor.output_stage_path("probe"),
            expected_inventory=inventory,
        )
        assert result_path.stat().st_mode & 0o777 == 0o444
    assert not (tmp_path / "canonical").exists()


@pytest.mark.parametrize("attempt", range(3))
def test_real_docker_oom_is_positive_or_fails_closed_without_inventing_timeout(
    tmp_path: Path,
    attempt: int,
) -> None:
    executor, source, code, _, _ = _executor(
        tmp_path,
        owner_id=f"item12-real-oom-{attempt}",
        mode="oom",
        wall_seconds=20,
    )
    observation = executor.run(
        execution_id="probe",
        source_path=source,
        code_path=code,
        output_path=executor.output_stage_path("probe"),
    )
    assert observation.exit_code == 137
    assert not observation.timed_out
    assert not observation.worker_watchdog_timed_out
    if observation.oom_killed:
        assert observation.outcome == "oom_killed"
    else:
        assert observation.outcome == "worker_failure"
    assert observation.container_removed
    assert observation.snapshot_volume_removed
    assert observation.process_tree_cleaned
    assert not observation.canonical_publication_performed
    assert not (tmp_path / "canonical").exists()


@pytest.mark.parametrize("attempt", range(3))
def test_real_docker_worker_watchdog_exit_is_typed_separately_from_parent_timeout(
    tmp_path: Path,
    attempt: int,
) -> None:
    executor, source, code, _, _ = _executor(
        tmp_path,
        owner_id=f"item12-real-worker-watchdog-{attempt}",
        mode="sleep-tree",
        wall_seconds=10,
    )
    observation = executor.run(
        execution_id="probe",
        source_path=source,
        code_path=code,
        output_path=executor.output_stage_path("probe"),
    )
    assert observation.outcome == "timeout"
    assert observation.exit_code == 124
    assert not observation.timed_out
    assert observation.worker_watchdog_timed_out
    assert not observation.oom_killed
    assert observation.container_removed
    assert observation.snapshot_volume_removed


def test_real_docker_worker_accepts_only_the_frozen_contained_topology(tmp_path: Path) -> None:
    source = tmp_path / "source-free-topology-probe.h5ad"
    source.write_bytes(b"topology-only; worker must not open this fixture\n")
    output = tmp_path / "output"
    output.mkdir()
    probe = """
from pathlib import Path
from scripts.sciplex3_k562_v5_worker import _require_exact_contained_invocation
from cellstate.backends.sciplex3_loader import SCIPLEX3_SOURCE_BYTE_COUNT, SCIPLEX3_SOURCE_SHA256
_require_exact_contained_invocation(
    source_path=Path('/run/cellstate/source/source.h5ad'),
    output_path=Path('/run/cellstate/output'),
    repository_root=Path('/workspace'),
    execution_id='sciplex3-k562-v5-fit',
    expected_source_sha256=SCIPLEX3_SOURCE_SHA256,
    expected_source_byte_count=SCIPLEX3_SOURCE_BYTE_COUNT,
    snapshot_directory=Path('/run/cellstate/snapshot'),
    snapshot_max_bytes=3 * 1024**3,
)
print('contained-topology-verified')
"""
    completed = subprocess.run(
        (
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "--pull",
            "never",
            "--memory",
            str(4 * 1024**3),
            "--memory-swap",
            str(4 * 1024**3),
            "--pids-limit",
            "256",
            "--user",
            f"{os.geteuid()}:{os.getegid()}",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--init",
            "--tmpfs",
            (
                "/run/cellstate/tmp:rw,noexec,nosuid,nodev,size=268435456,mode=0700,"
                f"uid={os.geteuid()},gid={os.getegid()}"
            ),
            "--mount",
            f"type=bind,source={source},target=/run/cellstate/source/source.h5ad,readonly",
            "--mount",
            f"type=bind,source={REPOSITORY_ROOT},target=/workspace,readonly",
            "--mount",
            f"type=bind,source={output},target=/run/cellstate/output",
            "--mount",
            "type=volume,target=/run/cellstate/snapshot",
            "--workdir",
            "/workspace",
            "--env",
            "PYTHONPATH=/workspace/src:/workspace",
            "--entrypoint",
            "/opt/runtime/bin/python",
            IMAGE_REFERENCE,
            "-c",
            probe,
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "contained-topology-verified"
    assert source.read_bytes() == b"topology-only; worker must not open this fixture\n"


_CHILD = r"""
import sys
from pathlib import Path
from cellstate.training.execution import (
    ContainedExecutionPolicy, DockerExecutor, ExecutionInputClosureManifest,
)
policy = ContainedExecutionPolicy.model_validate_json(Path(sys.argv[1]).read_bytes())
inputs = ExecutionInputClosureManifest.model_validate_json(Path(sys.argv[2]).read_bytes())
root = Path(sys.argv[3])
executor = DockerExecutor(
    policy,
    lock_root=root / "locks",
    staging_root=root / "stage",
    canonical_publication_root=root / "canonical",
    execution_input_closure=inputs,
)
executor.run(
    execution_id="death",
    source_path=root / "source.h5ad",
    code_path=root / "code",
    output_path=executor.output_stage_path("death"),
)
"""


def test_worker_watchdog_bounds_supervisor_death_and_new_policy_removes_volume(
    tmp_path: Path,
) -> None:
    _, _, _, policy, inputs = _executor(
        tmp_path,
        owner_id="item12-real-parent-death",
        mode="sleep-tree",
        wall_seconds=30,
    )
    policy_path = tmp_path / "policy.json"
    input_path = tmp_path / "inputs.json"
    policy_path.write_bytes(canonical_json_bytes(policy.model_dump(mode="json")))
    input_path.write_bytes(canonical_json_bytes(inputs.model_dump(mode="json")))
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(policy_path), str(input_path), str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "src")},
    )
    container_id = ""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        listed = subprocess.run(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label=org.cellstate.contained-execution.owner={policy.owner_id}",
                "--filter",
                f"label=org.cellstate.contained-execution.policy-sha256={policy.fingerprint}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if listed.stdout.strip():
            container_id = listed.stdout.strip()
            break
        time.sleep(0.1)
    if not container_id:
        stdout, stderr = child.communicate(timeout=10)
        pytest.fail(
            "supervisor exited before creating its labeled container: "
            f"stdout={stdout!r}, stderr={stderr!r}"
        )
    mounts = json.loads(
        subprocess.run(
            ["docker", "inspect", "--format", "{{json .Mounts}}", container_id],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    volume_id = next(
        mount["Name"]
        for mount in mounts
        if mount["Type"] == "volume" and mount["Destination"] == policy.snapshot_container_path
    )
    os.kill(child.pid, signal.SIGKILL)
    child.wait(timeout=10)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = json.loads(
            subprocess.run(
                ["docker", "inspect", "--format", "{{json .State}}", container_id],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        if state["Running"] is False:
            break
        time.sleep(0.1)
    else:
        pytest.fail("in-container watchdog did not bound the orphaned worker")

    next_policy_payload = policy.model_dump(mode="python")
    next_policy_payload["policy_id"] = f"{policy.policy_id}-next"
    next_policy = ContainedExecutionPolicy.model_validate(next_policy_payload)
    assert next_policy.fingerprint != policy.fingerprint
    next_executor = DockerExecutor(
        next_policy,
        lock_root=tmp_path / "locks",
        staging_root=tmp_path / "stage",
        canonical_publication_root=tmp_path / "canonical",
        execution_input_closure=inputs,
    )
    assert next_executor.recover_owned_containers() == (container_id,)
    assert (
        subprocess.run(
            ["docker", "volume", "inspect", volume_id],
            check=False,
            capture_output=True,
            text=True,
        ).returncode
        != 0
    )
