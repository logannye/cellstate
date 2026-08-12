"""Source-free terminal-evidence and supervisor tests for Item 12.3."""

from __future__ import annotations

import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import ModuleType, SimpleNamespace
from typing import Literal, cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cellstate.evaluation as evaluation_package
import scripts.run_sciplex3_k562_v5_contained as supervisor
import scripts.sciplex3_k562_v5_worker as worker
from cellstate.backends.sciplex3_loader import (
    SCIPLEX3_SOURCE_BYTE_COUNT,
    SCIPLEX3_SOURCE_SHA256,
)
from cellstate.domain.common import canonical_json_bytes
from cellstate.training.execution import (
    PARENT_TERMINAL_REPORT_MAX_BYTES,
    WORKER_TERMINAL_REPORT_MAX_BYTES,
    ContainedExecutionError,
    ContainedExecutionObservation,
    ContainedTrainingTerminalObservation,
    ContainedTrainingWorkerObservation,
    ContainedTrainingWorkerTerminalReport,
    DockerExecutor,
    ExecutionStageAlreadyClaimed,
    canonical_publication_tree_identity,
    inventory_staged_training_tree,
)
from cellstate.training.item12_3_authorization import (
    Item123AuthorizationError,
    Item123ExecutionStart,
    VerifiedItem123ExecutionCapability,
)

_IMAGE_DIGEST = "sha256:" + "a" * 64
_ExecutionOutcome = Literal["success", "timeout", "oom_killed", "worker_failure"]
_FAKE_EXECUTION_START = cast(Item123ExecutionStart, SimpleNamespace())


def _contracts() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    policy = SimpleNamespace(
        owner_id="sciplex3-k562-v5",
        fingerprint="1" * 64,
        runtime_image=SimpleNamespace(digest=_IMAGE_DIGEST),
        container_user_mode="host-effective-uid-gid",
        wall_clock_seconds=3_600,
        memory_max_bytes=4 * 1024**3,
        memory_swap_max_bytes=4 * 1024**3,
    )
    return (
        policy,
        SimpleNamespace(fingerprint="2" * 64),
        SimpleNamespace(fingerprint="3" * 64),
        SimpleNamespace(runtime_image=SimpleNamespace(digest=_IMAGE_DIGEST)),
    )


def _execution(
    policy: SimpleNamespace, outcome: _ExecutionOutcome
) -> ContainedExecutionObservation:
    timed_out = outcome == "timeout"
    oom_killed = outcome == "oom_killed"
    exit_code = 0 if outcome == "success" else 137 if timed_out or oom_killed else 2
    return ContainedExecutionObservation(
        execution_id=supervisor.CONTAINED_EXECUTION_ID,
        policy_fingerprint=policy.fingerprint,
        runtime_image_digest=policy.runtime_image.digest,
        container_user_mode="host-effective-uid-gid",
        observed_container_uid=1000,
        observed_container_gid=1000,
        outcome=outcome,
        exit_code=exit_code,
        timed_out=timed_out,
        oom_killed=oom_killed,
        parent_wall_clock_elapsed_seconds=10.0,
    )


def _populate_success_stage(
    output: Path,
    *,
    contracts: tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, SimpleNamespace],
) -> None:
    policy, code_closure, input_closure, image_lock = contracts
    for relative_path in supervisor._SUCCESS_STAGE_PATHS:
        path = output / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative_path in {
            "candidate-model.json",
            "fit/candidate-model.json",
        }:
            payload = b'{"model":true}'
        elif relative_path in {
            "candidate-training-plan.json",
            "sealed-plan/candidate-training-plan.json",
        }:
            payload = b'{"plan":true}'
        elif relative_path in {
            "training-execution-observation.json",
            "fit/training-execution-observation.json",
        }:
            payload = b'{"observation":true}'
        else:
            payload = canonical_json_bytes({"path": relative_path})
        path.write_bytes(payload)
    inventory = inventory_staged_training_tree(output)
    success = ContainedTrainingWorkerObservation(
        execution_id=supervisor.CONTAINED_EXECUTION_ID,
        training_plan_fingerprint="4" * 64,
        policy_fingerprint=policy.fingerprint,
        runtime_image_digest=image_lock.runtime_image.digest,
        training_code_closure_sha256=code_closure.fingerprint,
        execution_input_closure_sha256=input_closure.fingerprint,
        expected_source_sha256=SCIPLEX3_SOURCE_SHA256,
        source_pre_sha256=SCIPLEX3_SOURCE_SHA256,
        source_post_sha256=SCIPLEX3_SOURCE_SHA256,
        expected_source_byte_count=SCIPLEX3_SOURCE_BYTE_COUNT,
        source_pre_byte_count=SCIPLEX3_SOURCE_BYTE_COUNT,
        source_post_byte_count=SCIPLEX3_SOURCE_BYTE_COUNT,
        staged_inventory=inventory,
        staged_tree_sha256=inventory.fingerprint,
        staged_file_count=len(inventory.entries),
    )
    report = ContainedTrainingWorkerTerminalReport(
        execution_id=supervisor.CONTAINED_EXECUTION_ID,
        outcome="success",
        terminal_phase="completed",
        failure_code="none",
        expected_source_sha256=SCIPLEX3_SOURCE_SHA256,
        expected_source_byte_count=SCIPLEX3_SOURCE_BYTE_COUNT,
        source_pre_sha256=SCIPLEX3_SOURCE_SHA256,
        source_pre_byte_count=SCIPLEX3_SOURCE_BYTE_COUNT,
        source_post_sha256=SCIPLEX3_SOURCE_SHA256,
        source_post_byte_count=SCIPLEX3_SOURCE_BYTE_COUNT,
        source_access_started=True,
        protected_source_acquired=True,
        source_post_authentication_attempted=True,
        source_post_authentication_completed=True,
        source_matches_expected=True,
        staged_inventory_status="verified",
        staged_tree_sha256=inventory.fingerprint,
        staged_file_count=len(inventory.entries),
        success_observation=success,
    )
    (output / supervisor.WORKER_TERMINAL_REPORT_FILENAME).write_bytes(
        canonical_json_bytes(report.model_dump(mode="json"))
    )


def _run_supervisor_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    scenario: str,
) -> tuple[Path, ContainedTrainingTerminalObservation, Path]:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    staging_root = tmp_path / "staging"
    contracts = _contracts()
    policy = contracts[0]

    class FakeExecutor:
        def __init__(self, *args: object, staging_root: Path, **kwargs: object) -> None:
            del args, kwargs
            self.staging_root = staging_root
            self.claimed_output: Path | None = None

        def output_stage_path(self, execution_id: str) -> Path:
            return self.staging_root / policy.owner_id / execution_id / "output"

        def run(self, *, output_path: Path, **kwargs: object) -> ContainedExecutionObservation:
            del kwargs
            output_path.mkdir(parents=True)
            self.claimed_output = output_path
            if scenario in {
                "preflight_error",
                "create_error",
                "start_error",
                "wait_error",
                "cleanup_error",
            }:
                raise ContainedExecutionError(f"sensitive-{scenario}-detail")
            outcome = {
                "timeout": "timeout",
                "oom": "oom_killed",
                "worker_failure": "worker_failure",
            }.get(scenario, "success")
            result = _execution(policy, cast(_ExecutionOutcome, outcome))
            if scenario in {"success", "semantic_tamper"}:
                _populate_success_stage(output_path, contracts=contracts)
            elif scenario == "invalid_worker_report":
                (output_path / supervisor.WORKER_TERMINAL_REPORT_FILENAME).write_bytes(
                    b"sensitive-unbounded-worker-error"
                )
            if scenario == "canonical_tamper":
                canonical = repository_root / supervisor.CANONICAL_PUBLICATION_RELATIVE_PATH
                canonical.mkdir(parents=True)
                (canonical / "external-tamper.json").write_bytes(b"external")
            return result

        def owns_output_stage_claim(self, *, execution_id: str, output_path: Path) -> bool:
            del execution_id
            return self.claimed_output == output_path

    monkeypatch.setattr(supervisor, "contained_training_contracts", lambda root: contracts)
    monkeypatch.setattr(supervisor, "DockerExecutor", FakeExecutor)
    capability = SimpleNamespace(
        proposal=SimpleNamespace(
            execution_paths=SimpleNamespace(
                execution_id=supervisor.CONTAINED_EXECUTION_ID,
                staging_root=str(staging_root),
                protected_source_path=str(tmp_path / "unopened-protected-source.h5ad"),
                canonical_publication_root=(
                    supervisor.CANONICAL_PUBLICATION_RELATIVE_PATH.as_posix()
                ),
            )
        )
    )
    monkeypatch.setattr(
        supervisor,
        "verify_capability_for_execution",
        lambda candidate, repository_root: SimpleNamespace(),
    )
    monkeypatch.setattr(
        supervisor,
        "verify_execution_start",
        lambda capability, execution_start: None,
    )
    if scenario == "semantic_tamper":

        def reject_semantics(*args: object, **kwargs: object) -> str:
            del args, kwargs
            raise ContainedExecutionError("semantic stage tamper")

        monkeypatch.setattr(supervisor, "_semantic_stage_fingerprint", reject_semantics)
    else:
        monkeypatch.setattr(
            supervisor,
            "_semantic_stage_fingerprint",
            lambda *args, **kwargs: "5" * 64,
        )
    stage, terminal = supervisor._execute_contained_training(
        capability=cast(VerifiedItem123ExecutionCapability, capability),
        execution_start=_FAKE_EXECUTION_START,
        source_path=tmp_path / "unopened-protected-source.h5ad",
        repository_root=repository_root,
        protected_source_acquired=True,
    )
    return stage, terminal, repository_root


def test_worker_reauthenticates_private_descriptor_and_sanitizes_scientific_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source-free-fixture.h5ad"
    source.write_bytes(b"source-free-worker-fixture")
    output = tmp_path / "output"
    snapshot = tmp_path / "snapshot"
    output.mkdir()
    snapshot.mkdir()
    expected_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(worker, "_require_exact_contained_invocation", lambda **kwargs: None)

    fake_materializer = ModuleType("scripts.materialize_sciplex3_k562_p1_candidate")

    def fail_preparation(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("sensitive-scientific-failure")

    fake_materializer._prepare_exact_p1 = fail_preparation  # type: ignore[attr-defined]
    fake_runner = ModuleType("cellstate.evaluation.sciplex3_candidate_runner")
    fake_runner.contained_training_contracts = lambda root: (  # type: ignore[attr-defined]
        object(),
        object(),
        object(),
        object(),
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.materialize_sciplex3_k562_p1_candidate",
        fake_materializer,
    )
    monkeypatch.setattr(evaluation_package, "sciplex3_candidate_runner", fake_runner)
    real_hash_descriptor = worker._hash_descriptor
    hash_calls = 0

    def counted_hash(descriptor: int) -> tuple[str, int]:
        nonlocal hash_calls
        hash_calls += 1
        return real_hash_descriptor(descriptor)

    monkeypatch.setattr(worker, "_hash_descriptor", counted_hash)
    with pytest.raises(worker._ReportedWorkerFailure):
        worker.run_worker(
            source_path=source,
            output_path=output,
            repository_root=tmp_path,
            execution_id=supervisor.CONTAINED_EXECUTION_ID,
            expected_source_sha256=expected_sha256,
            expected_source_byte_count=source.stat().st_size,
            snapshot_directory=snapshot,
            snapshot_max_bytes=1024,
        )
    payload = (output / supervisor.WORKER_TERMINAL_REPORT_FILENAME).read_bytes()
    report = ContainedTrainingWorkerTerminalReport.model_validate_json(payload)
    assert hash_calls == 2
    assert report.outcome == "worker_failure"
    assert report.terminal_phase == "preparation"
    assert report.failure_code == "preparation_failed"
    assert report.source_pre_sha256 == report.source_post_sha256 == expected_sha256
    assert report.source_post_authentication_completed
    assert report.exception_detail_recorded is False
    assert b"sensitive" not in payload and b"RuntimeError" not in payload
    assert len(payload) <= WORKER_TERMINAL_REPORT_MAX_BYTES


@pytest.mark.parametrize(
    ("failure_point", "access_started", "post_authentication_attempted"),
    (
        ("source_open", False, False),
        ("snapshot_open", True, False),
        ("source_read", True, True),
        ("snapshot_write", True, True),
        ("snapshot_fsync", True, True),
    ),
)
def test_snapshot_failures_emit_conservative_bounded_source_acquisition_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    access_started: bool,
    post_authentication_attempted: bool,
) -> None:
    source = tmp_path / "source-free-snapshot-fixture.h5ad"
    source.write_bytes(b"source-free-snapshot-fixture")
    output = tmp_path / "output"
    snapshot = tmp_path / "snapshot"
    output.mkdir()
    snapshot.mkdir()
    expected_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(worker, "_require_exact_contained_invocation", lambda **kwargs: None)
    real_open = worker.os.open
    real_read = worker.os.read
    real_write = worker.os.write
    real_fsync = worker.os.fsync
    source_descriptor: int | None = None
    snapshot_descriptor: int | None = None
    injected = False

    def controlled_open(path: object, flags: int, mode: int = 0o777) -> int:
        nonlocal source_descriptor, snapshot_descriptor, injected
        candidate = Path(cast(str, path))
        if failure_point == "source_open" and candidate == source and not injected:
            injected = True
            raise OSError("sensitive injected source-open failure")
        if (
            failure_point == "snapshot_open"
            and candidate.name == ".contained-source-snapshot"
            and flags & worker.os.O_CREAT
            and not injected
        ):
            injected = True
            raise OSError("sensitive injected snapshot-open failure")
        descriptor = real_open(path, flags, mode)  # type: ignore[arg-type]
        if candidate == source:
            source_descriptor = descriptor
        elif candidate.name == ".contained-source-snapshot" and flags & worker.os.O_CREAT:
            snapshot_descriptor = descriptor
        return descriptor

    def controlled_read(descriptor: int, byte_count: int) -> bytes:
        nonlocal injected
        if failure_point == "source_read" and descriptor == source_descriptor and not injected:
            injected = True
            raise OSError("sensitive injected source-read failure")
        return real_read(descriptor, byte_count)

    def controlled_write(descriptor: int, payload: bytes) -> int:
        nonlocal injected
        if failure_point == "snapshot_write" and descriptor == snapshot_descriptor and not injected:
            injected = True
            raise OSError("sensitive injected snapshot-write failure")
        return real_write(descriptor, payload)

    def controlled_fsync(descriptor: int) -> None:
        nonlocal injected
        if failure_point == "snapshot_fsync" and descriptor == snapshot_descriptor and not injected:
            injected = True
            raise OSError("sensitive injected snapshot-fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(worker.os, "open", controlled_open)
    monkeypatch.setattr(worker.os, "read", controlled_read)
    monkeypatch.setattr(worker.os, "write", controlled_write)
    monkeypatch.setattr(worker.os, "fsync", controlled_fsync)
    with pytest.raises(worker._ReportedWorkerFailure):
        worker.run_worker(
            source_path=source,
            output_path=output,
            repository_root=tmp_path,
            execution_id=supervisor.CONTAINED_EXECUTION_ID,
            expected_source_sha256=expected_sha256,
            expected_source_byte_count=source.stat().st_size,
            snapshot_directory=snapshot,
            snapshot_max_bytes=1024,
        )
    assert injected
    payload = (output / supervisor.WORKER_TERMINAL_REPORT_FILENAME).read_bytes()
    report = ContainedTrainingWorkerTerminalReport.model_validate_json(payload)
    assert report.outcome == "worker_failure"
    assert report.failure_code == "source_authentication_failed"
    assert report.source_access_started is access_started
    assert report.protected_source_acquired is access_started
    assert report.source_post_authentication_attempted is post_authentication_attempted
    assert report.source_post_authentication_completed is post_authentication_attempted
    assert report.source_matches_expected is False
    assert report.source_pre_sha256 is None
    assert b"sensitive" not in payload
    assert len(payload) <= WORKER_TERMINAL_REPORT_MAX_BYTES


def test_raw_worker_rejects_arbitrary_paths_before_source_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_started = False

    def forbidden_snapshot(*args: object, **kwargs: object) -> object:
        nonlocal snapshot_started
        del args, kwargs
        snapshot_started = True
        raise AssertionError("raw invocation reached protected source snapshot")

    monkeypatch.setattr(worker, "_snapshot_pinned_source", forbidden_snapshot)
    with pytest.raises(RuntimeError, match="approved contained contract"):
        worker.run_worker(
            source_path=tmp_path / "arbitrary-source.h5ad",
            output_path=tmp_path / "arbitrary-output",
            repository_root=tmp_path,
            execution_id=supervisor.CONTAINED_EXECUTION_ID,
            expected_source_sha256=SCIPLEX3_SOURCE_SHA256,
            expected_source_byte_count=SCIPLEX3_SOURCE_BYTE_COUNT,
            snapshot_directory=tmp_path / "arbitrary-snapshot",
            snapshot_max_bytes=3 * 1024**3,
        )
    assert not snapshot_started


def test_canonical_publication_identity_detects_content_and_rejects_links(
    tmp_path: Path,
) -> None:
    root = tmp_path / "canonical"
    absent = canonical_publication_tree_identity(root)
    assert absent.state == "absent"
    root.mkdir()
    (root / "current.json").write_bytes(b"one")
    first = canonical_publication_tree_identity(root)
    (root / "current.json").write_bytes(b"two")
    second = canonical_publication_tree_identity(root)
    assert first.state == second.state == "present"
    assert first.tree_sha256 != second.tree_sha256
    (root / "linked.json").symlink_to(root / "current.json")
    with pytest.raises(ContainedExecutionError, match="regular file"):
        canonical_publication_tree_identity(root)


def test_success_is_semantically_verified_sealed_and_never_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, terminal, repository_root = _run_supervisor_fixture(
        tmp_path, monkeypatch, scenario="success"
    )
    assert terminal.terminal_status == "success"
    assert terminal.execution_observation is not None
    assert terminal.execution_observation.parent_wall_clock_elapsed_seconds == 10.0
    assert terminal.container_cleanup_disposition == "proved_removed"
    assert terminal.snapshot_volume_cleanup_disposition == "proved_removed"
    assert terminal.semantic_stage_sha256 == "5" * 64
    assert terminal.canonical_publication_unchanged
    assert stage.name == "output" and stage.stat().st_mode & 0o777 == 0o555
    assert not (repository_root / supervisor.CANONICAL_PUBLICATION_RELATIVE_PATH).exists()
    terminal_path = stage.parent / supervisor.PARENT_TERMINAL_REPORT_FILENAME
    assert terminal_path.is_file()
    assert terminal_path.stat().st_size <= PARENT_TERMINAL_REPORT_MAX_BYTES
    assert (
        ContainedTrainingTerminalObservation.model_validate_json(terminal_path.read_bytes())
        == terminal
    )
    with pytest.raises(ValueError, match="cleanup disposition"):
        ContainedTrainingTerminalObservation.model_validate(
            {
                **terminal.model_dump(mode="python"),
                "container_cleanup_disposition": "unproved",
            }
        )


@pytest.mark.parametrize(
    ("scenario", "status", "failure_code"),
    (
        ("timeout", "timeout", "worker_timed_out"),
        ("oom", "oom_killed", "worker_oom_killed"),
        ("worker_failure", "worker_failure", "worker_exited_nonzero"),
    ),
)
def test_docker_terminal_failures_are_durably_quarantined_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    status: str,
    failure_code: str,
) -> None:
    stage, terminal, repository_root = _run_supervisor_fixture(
        tmp_path, monkeypatch, scenario=scenario
    )
    assert terminal.terminal_status == status
    assert terminal.failure_code == failure_code
    assert terminal.protected_source_acquired_before_supervisor is True
    assert terminal.container_cleanup_disposition == "proved_removed"
    assert terminal.snapshot_volume_cleanup_disposition == "proved_removed"
    assert terminal.stage_disposition == "quarantined"
    assert stage.name == "quarantine" and not (stage.parent / "output").exists()
    assert (stage.parent / supervisor.PARENT_TERMINAL_REPORT_FILENAME).is_file()
    assert not (repository_root / supervisor.CANONICAL_PUBLICATION_RELATIVE_PATH).exists()


@pytest.mark.parametrize(
    "scenario",
    ("preflight_error", "create_error", "start_error", "wait_error", "cleanup_error"),
)
def test_executor_exceptions_get_sanitized_universal_evidence_after_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    stage, terminal, repository_root = _run_supervisor_fixture(
        tmp_path, monkeypatch, scenario=scenario
    )
    assert terminal.terminal_status == "supervisor_failure"
    assert terminal.failure_code == "contained_executor_failed"
    assert terminal.execution_observation is None
    assert terminal.aggregate_container_limits_enforced is False
    assert terminal.container_cleanup_disposition == "unproved"
    assert terminal.snapshot_volume_cleanup_disposition == "unproved"
    assert terminal.protected_source_acquired_before_supervisor is True
    assert stage.name == "quarantine"
    payload = (stage.parent / supervisor.PARENT_TERMINAL_REPORT_FILENAME).read_bytes()
    assert len(payload) <= PARENT_TERMINAL_REPORT_MAX_BYTES
    assert b"sensitive" not in payload and scenario.encode() not in payload
    assert not (repository_root / supervisor.CANONICAL_PUBLICATION_RELATIVE_PATH).exists()


def test_owner_lock_failure_after_real_executor_stage_claim_gets_parent_terminal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    staging_root = tmp_path / "staging"
    protected_source = tmp_path / "approved-but-unopened-source.h5ad"
    capability = SimpleNamespace(
        proposal=SimpleNamespace(
            execution_paths=SimpleNamespace(
                execution_id=supervisor.CONTAINED_EXECUTION_ID,
                staging_root=str(staging_root),
                protected_source_path=str(protected_source),
                canonical_publication_root=(
                    supervisor.CANONICAL_PUBLICATION_RELATIVE_PATH.as_posix()
                ),
            )
        )
    )
    monkeypatch.setattr(
        supervisor,
        "verify_capability_for_execution",
        lambda candidate, repository_root: SimpleNamespace(),
    )
    monkeypatch.setattr(
        supervisor,
        "verify_execution_start",
        lambda capability, execution_start: None,
    )

    class FailingOwnerLock:
        def __enter__(self) -> None:
            raise ContainedExecutionError("sensitive owner-lock failure")

        def __exit__(self, *args: object) -> None:
            del args

    monkeypatch.setattr(
        supervisor.DockerExecutor,
        "_owner_lock",
        lambda executor: FailingOwnerLock(),
    )
    stage, terminal = supervisor._execute_contained_training(
        capability=cast(VerifiedItem123ExecutionCapability, capability),
        execution_start=_FAKE_EXECUTION_START,
        source_path=protected_source,
        repository_root=repository_root,
        protected_source_acquired=True,
    )
    assert terminal.terminal_status == "supervisor_failure"
    assert terminal.failure_code == "contained_executor_failed"
    assert terminal.execution_observation is None
    assert terminal.container_cleanup_disposition == "unproved"
    assert terminal.snapshot_volume_cleanup_disposition == "unproved"
    assert stage.name == "quarantine"
    payload = (stage.parent / supervisor.PARENT_TERMINAL_REPORT_FILENAME).read_bytes()
    assert b"sensitive" not in payload
    assert not protected_source.exists()


def test_concurrent_supervisors_have_one_stage_owner_and_loser_cannot_quarantine_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    staging_root = tmp_path / "staging"
    protected_source = tmp_path / "approved-but-unopened-source.h5ad"
    capability = SimpleNamespace(
        proposal=SimpleNamespace(
            execution_paths=SimpleNamespace(
                execution_id=supervisor.CONTAINED_EXECUTION_ID,
                staging_root=str(staging_root),
                protected_source_path=str(protected_source),
                canonical_publication_root=(
                    supervisor.CANONICAL_PUBLICATION_RELATIVE_PATH.as_posix()
                ),
            )
        )
    )
    monkeypatch.setattr(
        supervisor,
        "verify_capability_for_execution",
        lambda candidate, repository_root: SimpleNamespace(),
    )
    monkeypatch.setattr(
        supervisor,
        "verify_execution_start",
        lambda capability, execution_start: None,
    )
    start_barrier = Barrier(2)

    class ConcurrentExecutor(DockerExecutor):
        def run(
            self,
            *,
            execution_id: str,
            source_path: Path,
            code_path: Path,
            output_path: Path,
        ) -> ContainedExecutionObservation:
            start_barrier.wait(timeout=5)
            return super().run(
                execution_id=execution_id,
                source_path=source_path,
                code_path=code_path,
                output_path=output_path,
            )

        def _run_contained(
            self,
            *,
            execution_id: str,
            source_path: Path,
            code_path: Path,
            output_path: Path,
        ) -> ContainedExecutionObservation:
            del source_path, code_path
            self._authenticate_prepared_output_stage(
                execution_id=execution_id,
                output_path=output_path,
            )
            (output_path / "winner-sentinel.txt").write_bytes(b"winner-owned-stage")
            return ContainedExecutionObservation(
                execution_id=execution_id,
                policy_fingerprint=self.policy.fingerprint,
                runtime_image_digest=self.policy.runtime_image.digest,
                container_user_mode="host-effective-uid-gid",
                observed_container_uid=1000,
                observed_container_gid=1000,
                outcome="worker_failure",
                exit_code=2,
                timed_out=False,
                oom_killed=False,
                parent_wall_clock_elapsed_seconds=1.0,
            )

    monkeypatch.setattr(supervisor, "DockerExecutor", ConcurrentExecutor)

    def invoke() -> object:
        try:
            return supervisor._execute_contained_training(
                capability=cast(VerifiedItem123ExecutionCapability, capability),
                execution_start=_FAKE_EXECUTION_START,
                source_path=protected_source,
                repository_root=repository_root,
                protected_source_acquired=True,
            )
        except BaseException as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(invoke), pool.submit(invoke))
        outcomes = tuple(future.result() for future in futures)
    losers = tuple(item for item in outcomes if isinstance(item, ExecutionStageAlreadyClaimed))
    winners = tuple(item for item in outcomes if isinstance(item, tuple))
    assert len(losers) == len(winners) == 1
    stage, terminal = cast(tuple[Path, ContainedTrainingTerminalObservation], winners[0])
    assert stage.name == "quarantine"
    assert terminal.terminal_status == "worker_failure"
    assert (stage / "winner-sentinel.txt").read_bytes() == b"winner-owned-stage"
    assert (stage.parent / supervisor.PARENT_TERMINAL_REPORT_FILENAME).is_file()
    assert not (stage.parent / "output").exists()


@pytest.mark.parametrize(
    ("scenario", "failure_code"),
    (
        ("invalid_worker_report", "worker_report_invalid"),
        ("semantic_tamper", "stage_semantic_verification_failed"),
    ),
)
def test_worker_or_semantic_stage_tamper_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    failure_code: str,
) -> None:
    stage, terminal, repository_root = _run_supervisor_fixture(
        tmp_path, monkeypatch, scenario=scenario
    )
    assert terminal.terminal_status == "stage_rejected"
    assert terminal.failure_code == failure_code
    assert stage.name == "quarantine"
    assert not (repository_root / supervisor.CANONICAL_PUBLICATION_RELATIVE_PATH).exists()


def test_canonical_publication_post_identity_detects_external_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, terminal, repository_root = _run_supervisor_fixture(
        tmp_path, monkeypatch, scenario="canonical_tamper"
    )
    canonical = repository_root / supervisor.CANONICAL_PUBLICATION_RELATIVE_PATH
    assert terminal.terminal_status == "stage_rejected"
    assert terminal.failure_code == "canonical_publication_changed"
    assert terminal.canonical_publication_before.state == "absent"
    assert terminal.canonical_publication_after is not None
    assert terminal.canonical_publication_after.state == "present"
    assert terminal.canonical_publication_unchanged is False
    assert stage.name == "quarantine"
    assert tuple(path.name for path in canonical.iterdir()) == ("external-tamper.json",)


def test_capability_denial_happens_before_private_source_coercion_and_raw_cli_is_disabled(
    tmp_path: Path,
) -> None:
    class SourcePathTrap:
        touched = False

        def __fspath__(self) -> str:
            self.touched = True
            raise AssertionError("denied execution inspected the source locator")

    with pytest.raises(Item123AuthorizationError, match="exact verified capability type"):
        supervisor.run_contained_training(
            capability=object(),  # type: ignore[arg-type]
            execution_start=cast(Item123ExecutionStart, object()),
            repository_root=tmp_path,
            protected_source_acquired=True,
        )
    private_source = SourcePathTrap()
    with pytest.raises(Item123AuthorizationError, match="exact verified capability type"):
        supervisor._execute_contained_training(
            capability=object(),  # type: ignore[arg-type]
            execution_start=cast(Item123ExecutionStart, object()),
            source_path=cast(Path, private_source),
            repository_root=tmp_path,
            protected_source_acquired=True,
        )
    assert private_source.touched is False
    with pytest.raises(SystemExit) as error:
        supervisor.main([])
    assert error.value.code == 2


def test_execution_start_denial_happens_before_private_source_coercion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SourcePathTrap:
        touched = False

        def __fspath__(self) -> str:
            self.touched = True
            raise AssertionError("denied execution inspected the protected-source path")

    capability = cast(VerifiedItem123ExecutionCapability, SimpleNamespace())
    source = SourcePathTrap()
    monkeypatch.setattr(
        supervisor,
        "verify_capability_for_execution",
        lambda candidate, repository_root: SimpleNamespace(),
    )

    def deny_start(capability: object, execution_start: object) -> None:
        del capability, execution_start
        raise Item123AuthorizationError("execution-start receipt differs")

    monkeypatch.setattr(supervisor, "verify_execution_start", deny_start)
    with pytest.raises(Item123AuthorizationError, match="execution-start receipt differs"):
        supervisor._execute_contained_training(
            capability=capability,
            execution_start=_FAKE_EXECUTION_START,
            source_path=cast(Path, source),
            repository_root=tmp_path,
            protected_source_acquired=True,
        )
    assert not source.touched


def test_private_supervisor_rejects_alternate_source_before_contract_or_docker_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approved_source = tmp_path / "approved-protected-source.h5ad"
    alternate_source = tmp_path / "alternate-protected-source.h5ad"
    capability = SimpleNamespace(
        proposal=SimpleNamespace(
            execution_paths=SimpleNamespace(
                execution_id=supervisor.CONTAINED_EXECUTION_ID,
                staging_root=str(tmp_path / "staging"),
                protected_source_path=str(approved_source),
                canonical_publication_root=(
                    supervisor.CANONICAL_PUBLICATION_RELATIVE_PATH.as_posix()
                ),
            )
        )
    )
    monkeypatch.setattr(
        supervisor,
        "verify_capability_for_execution",
        lambda candidate, repository_root: SimpleNamespace(),
    )
    monkeypatch.setattr(
        supervisor,
        "verify_execution_start",
        lambda capability, execution_start: None,
    )
    contract_work_started = False

    def forbidden_contract_work(root: Path) -> object:
        nonlocal contract_work_started
        del root
        contract_work_started = True
        raise AssertionError("alternate source reached contract or Docker setup")

    monkeypatch.setattr(supervisor, "contained_training_contracts", forbidden_contract_work)
    with pytest.raises(ContainedExecutionError, match="differs from the verified capability"):
        supervisor._execute_contained_training(
            capability=cast(VerifiedItem123ExecutionCapability, capability),
            execution_start=_FAKE_EXECUTION_START,
            source_path=alternate_source,
            repository_root=tmp_path,
            protected_source_acquired=True,
        )
    assert not contract_work_started
    assert not approved_source.exists() and not alternate_source.exists()
