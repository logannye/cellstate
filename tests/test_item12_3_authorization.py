"""Source-free authorization, replay, wrapper, and workflow tests for Item 12.3."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Any, cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cellstate.training.item12_3_authorization as authorization
import scripts.run_item12_3_authorized_execution as authorized_runner
import scripts.run_sciplex3_k562_v5_contained as supervisor
import scripts.verify_item12_3_authorization as authorization_cli
from cellstate.domain.common import canonical_json_bytes
from cellstate.training.item12_3_authorization import (
    ITEM12_3_LEDGER_ROOT,
    ITEM12_3_WORKFLOW_DISPATCH_REF,
    Item123AuthorizationError,
    Item123DispatchApproval,
    Item123ReplayError,
    VerifiedItem123ExecutionCapability,
    build_durable_fallback_terminal,
    build_pending_proposal,
    canonical_capability_bytes,
    canonical_proposal_bytes,
    claim_execution_start,
    consume_attempt,
    durable_attempt_tag,
    durable_release_body,
    issue_verified_capability,
    load_approved_proposal,
    load_attempt_consumption,
    load_durable_consumption_observation,
    load_runtime_preparation_payload,
    load_verified_capability,
    record_contained_terminal_report,
    record_terminal_report,
    require_proposal_current,
    verify_capability_for_execution,
    verify_durable_release,
    verify_execution_start,
    verify_proposal_against_repository,
)

_ROOT = Path(__file__).resolve().parents[1]
_C = "a" * 40
_D = "b" * 40


@pytest.fixture(scope="module")
def proposal() -> authorization.Item123ExecutionProposal:
    return build_pending_proposal(
        _ROOT,
        execution_repository_commit=_C,
        not_before_utc="2026-08-11T00:00:00Z",
        expires_at_utc="2026-08-12T00:00:00Z",
    )


def _approval(
    proposal: authorization.Item123ExecutionProposal,
    *,
    run_id: int = 11,
    run_attempt: int = 1,
) -> Item123DispatchApproval:
    return Item123DispatchApproval(
        approved_proposal_sha256=proposal.fingerprint,
        dispatch_actor="logannye",
        triggering_actor="logannye",
        github_run_id=run_id,
        github_run_attempt=run_attempt,
        workflow_dispatch_ref=ITEM12_3_WORKFLOW_DISPATCH_REF,
        workflow_repository_commit=_C,
        proposal_repository_commit=_D,
    )


def _durable(
    proposal: authorization.Item123ExecutionProposal,
    approval: Item123DispatchApproval,
) -> authorization.Item123DurableReleaseObservation:
    tag = durable_attempt_tag(proposal.fingerprint)
    body = durable_release_body(proposal, approval, proposal_repository_commit=_D).decode()
    return verify_durable_release(
        proposal,
        approval,
        proposal_repository_commit=_D,
        release_payload={
            "id": 123,
            "tag_name": tag,
            "target_commitish": _D,
            "name": tag,
            "body": body,
            "draft": False,
            "prerelease": False,
            "immutable": True,
            "assets": [],
        },
        tag_ref_payload={"ref": f"refs/tags/{tag}", "object": {"type": "commit", "sha": _D}},
        gh_release_verify_succeeded=True,
        release_creation_succeeded_in_this_run=True,
    )


def _runtime_payload(proposal: authorization.Item123ExecutionProposal) -> bytes:
    distribution = proposal.runtime_distribution
    native = proposal.native_runtime
    return canonical_json_bytes(
        {
            "archive": {
                "archive_sha256": distribution.asset_sha256,
                "config_digest": distribution.config_digest,
                "image_digest": distribution.image_digest,
                "index_digest": distribution.oci_index_digest,
                "layers": [
                    layer.model_dump(mode="json")
                    for layer in proposal.contained_execution.runtime_layers
                ],
            },
            "archive_file": {
                "byte_count": distribution.asset_byte_count,
                "change_time_ns": 4,
                "device": 1,
                "inode": 2,
                "mode": 0o400,
                "modification_time_ns": 3,
            },
            "daemon": {
                "architecture": native.docker_architecture,
                "cgroup_version": native.cgroup_version,
                "context_name": "setup-docker-action",
                "endpoint": "unix:///tmp/docker.sock",
                "host_architecture": native.host_architecture,
                "host_operating_system": native.host_operating_system,
                "image_store_status": list(native.image_store_status),
                "memory_limit_supported": True,
                "memory_swap_limit_supported": True,
                "operating_system": native.docker_operating_system,
                "pids_limit_supported": True,
                "server_version": native.docker_server_version,
            },
            "distribution_lock_sha256": distribution.distribution_lock_sha256,
            "load_performed_from_verified_descriptor": True,
            "loaded_image_verified": True,
            "release": {
                "asset_attestation_verified": True,
                "asset_byte_count": distribution.asset_byte_count,
                "asset_name": distribution.asset_name,
                "asset_sha256": distribution.asset_sha256,
                "asset_url": distribution.qualified_asset_url,
                "attestation_predicate_type": distribution.attestation_predicate_type,
                "release_attestation_verified": True,
                "release_tag": distribution.release_tag,
                "release_target_commit": distribution.release_target_commit,
                "repository": distribution.repository,
            },
            "runtime_image_lock_sha256": distribution.runtime_image_lock_sha256,
        }
    )


def _fake_ledger(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    root.mkdir(mode=0o700)

    def inspect(_: Path) -> tuple[Path, int]:
        return root, os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))

    monkeypatch.setattr(authorization, "_inspect_ledger_root", inspect)
    monkeypatch.setattr(authorization, "require_proposal_current", lambda _: None)


def _capability(
    proposal: authorization.Item123ExecutionProposal,
    approval: Item123DispatchApproval,
    durable: authorization.Item123DurableReleaseObservation,
    consumption: authorization.Item123AttemptConsumption,
) -> VerifiedItem123ExecutionCapability:
    return issue_verified_capability(
        proposal, approval, durable, consumption, _runtime_payload(proposal)
    )


def _write_sealed(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o400)


def test_pending_proposal_is_canonical_source_free_and_discloses_scope(
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    assert hashlib.sha256(canonical_proposal_bytes(proposal)).hexdigest() == proposal.fingerprint
    assert proposal.grants_execution_authority is False
    assert proposal.protected_source.byte_count == 2_526_631_614
    assert proposal.protected_source.entire_opaque_asset_transfer_and_snapshot_required is True
    assert (
        proposal.protected_source.permitted_expression_or_raw_count_decode_scope
        == "p1-train-rows-only"
    )
    assert proposal.protected_source.full_axis_selector_metadata_decode_required is True
    assert proposal.protected_source.heldout_expression_or_raw_count_values_read is False
    assert proposal.protected_source.prohibited_partitions == (
        "p2-calibration",
        "p3-model-selection-validation",
        "p4-untouched-test",
    )
    assert len(proposal.authorization_tools) == 6
    assert not hasattr(authorization, "consume_before_source")
    assert "consume_before_source" not in authorization.__all__


def test_proposal_schema_rejects_authority_scope_and_closure_drift(
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    base = proposal.model_dump(mode="python")
    mutations: tuple[tuple[tuple[str, ...], object, str], ...] = (
        (("expires_at_utc",), "2026-08-12T00:00:01Z", "validity interval"),
        (("protected_source", "filename"), "other.h5ad", "exact p1 descriptor"),
        (("resource_limits", "memory_swap_max_bytes"), 1, "resource gates"),
        (
            ("execution_paths", "permitted_stage_relative_paths"),
            ("model.npz",),
            "stage output allowlist",
        ),
        (("candidate", "candidate_id"), "candidate-drift", "frozen v5 candidate"),
        (("runtime_distribution", "asset_sha256"), "0" * 64, "distribution joins"),
    )
    for path, value, match in mutations:
        candidate = deepcopy(base)
        target: dict[str, Any] = candidate
        for component in path[:-1]:
            target = cast(dict[str, Any], target[component])
        target[path[-1]] = value
        with pytest.raises(ValueError, match=match):
            authorization.Item123ExecutionProposal.model_validate(candidate)

    duplicate_tools = deepcopy(base)
    tools = list(duplicate_tools["authorization_tools"])
    tools[0] = tools[1]
    duplicate_tools["authorization_tools"] = tuple(tools)
    with pytest.raises(ValueError, match="tooling closure"):
        authorization.Item123ExecutionProposal.model_validate(duplicate_tools)
    with pytest.raises(ValueError, match="canonical repository-relative"):
        authorization.Item123ToolBinding(relative_path="../escape.py", sha256="0" * 64)
    with pytest.raises(ValueError, match="whole-second RFC3339"):
        authorization.Item123ExecutionProposal.model_validate(
            {**base, "not_before_utc": "2026-08-11T00:00:00.1Z"}
        )


def test_approved_proposal_round_trip_and_file_aliases_fail_closed(
    tmp_path: Path,
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    payload = canonical_proposal_bytes(proposal)
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_bytes(payload)
    observed, approval = load_approved_proposal(
        proposal_path,
        proposal.fingerprint,
        dispatch_actor="logannye",
        triggering_actor="logannye",
        github_run_id=17,
        github_run_attempt=1,
        workflow_dispatch_ref=ITEM12_3_WORKFLOW_DISPATCH_REF,
        workflow_repository_commit=_C,
        proposal_repository_commit=_D,
    )
    assert observed == proposal
    assert approval.approved_proposal_sha256 == proposal.fingerprint

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(payload + b"\n")
    with pytest.raises(Item123AuthorizationError, match="canonical JSON"):
        load_approved_proposal(
            noncanonical,
            hashlib.sha256(noncanonical.read_bytes()).hexdigest(),
            dispatch_actor="logannye",
            triggering_actor="logannye",
            github_run_id=17,
            github_run_attempt=1,
            workflow_dispatch_ref=ITEM12_3_WORKFLOW_DISPATCH_REF,
            workflow_repository_commit=_C,
            proposal_repository_commit=_D,
        )

    hardlink = tmp_path / "hardlink.json"
    os.link(proposal_path, hardlink)
    with pytest.raises(Item123AuthorizationError, match="non-hardlinked regular file"):
        load_approved_proposal(
            hardlink,
            proposal.fingerprint,
            dispatch_actor="logannye",
            triggering_actor="logannye",
            github_run_id=17,
            github_run_attempt=1,
            workflow_dispatch_ref=ITEM12_3_WORKFLOW_DISPATCH_REF,
            workflow_repository_commit=_C,
            proposal_repository_commit=_D,
        )

    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(proposal_path)
    with pytest.raises(Item123AuthorizationError, match="non-hardlinked regular file"):
        load_approved_proposal(
            symlink,
            proposal.fingerprint,
            dispatch_actor="logannye",
            triggering_actor="logannye",
            github_run_id=17,
            github_run_attempt=1,
            workflow_dispatch_ref=ITEM12_3_WORKFLOW_DISPATCH_REF,
            workflow_repository_commit=_C,
            proposal_repository_commit=_D,
        )


def test_raw_digest_actor_ref_and_validity_deny_before_other_repository_reads(
    tmp_path: Path,
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    path = tmp_path / "proposal.json"
    path.write_bytes(canonical_proposal_bytes(proposal))
    with pytest.raises(Item123AuthorizationError, match="approved digest"):
        load_approved_proposal(
            path,
            "0" * 64,
            dispatch_actor="logannye",
            triggering_actor="logannye",
            github_run_id=1,
            github_run_attempt=1,
            workflow_dispatch_ref=ITEM12_3_WORKFLOW_DISPATCH_REF,
            workflow_repository_commit=_C,
            proposal_repository_commit=_D,
        )
    with pytest.raises(Item123AuthorizationError, match="exact lowercase SHA-256"):
        load_approved_proposal(
            path,
            "NOT-A-DIGEST",
            dispatch_actor="logannye",
            triggering_actor="logannye",
            github_run_id=1,
            github_run_attempt=1,
            workflow_dispatch_ref=ITEM12_3_WORKFLOW_DISPATCH_REF,
            workflow_repository_commit=_C,
            proposal_repository_commit=_D,
        )
    with pytest.raises(Item123AuthorizationError, match="actor or trusted dispatch ref"):
        load_approved_proposal(
            path,
            proposal.fingerprint,
            dispatch_actor="mallory",
            triggering_actor="logannye",
            github_run_id=1,
            github_run_attempt=1,
            workflow_dispatch_ref=ITEM12_3_WORKFLOW_DISPATCH_REF,
            workflow_repository_commit=_C,
            proposal_repository_commit=_D,
        )
    with pytest.raises(Item123AuthorizationError, match="rerun attempts"):
        load_approved_proposal(
            path,
            proposal.fingerprint,
            dispatch_actor="logannye",
            triggering_actor="logannye",
            github_run_id=1,
            github_run_attempt=2,
            workflow_dispatch_ref=ITEM12_3_WORKFLOW_DISPATCH_REF,
            workflow_repository_commit=_C,
            proposal_repository_commit=_D,
        )
    with pytest.raises(Item123AuthorizationError, match="workflow commit differs"):
        load_approved_proposal(
            path,
            proposal.fingerprint,
            dispatch_actor="logannye",
            triggering_actor="logannye",
            github_run_id=1,
            github_run_attempt=1,
            workflow_dispatch_ref=ITEM12_3_WORKFLOW_DISPATCH_REF,
            workflow_repository_commit="c" * 40,
            proposal_repository_commit=_D,
        )
    with pytest.raises(Item123AuthorizationError, match="proposal D must be distinct"):
        load_approved_proposal(
            path,
            proposal.fingerprint,
            dispatch_actor="logannye",
            triggering_actor="logannye",
            github_run_id=1,
            github_run_attempt=1,
            workflow_dispatch_ref=ITEM12_3_WORKFLOW_DISPATCH_REF,
            workflow_repository_commit=_C,
            proposal_repository_commit=_C,
        )
    with pytest.raises(Item123AuthorizationError, match="not yet valid"):
        require_proposal_current(
            proposal,
            current_time_utc=authorization._parse_rfc3339_utc(
                "2026-08-10T23:59:59Z", name="test time"
            ),
        )
    with pytest.raises(Item123AuthorizationError, match="expired"):
        require_proposal_current(
            proposal,
            current_time_utc=authorization._parse_rfc3339_utc(
                "2026-08-12T00:00:00Z", name="test time"
            ),
        )
    with pytest.raises(Item123AuthorizationError, match="timezone-aware UTC"):
        require_proposal_current(proposal, current_time_utc=datetime(2026, 8, 11, 12))


def test_durable_release_binds_run_winner_and_embeds_unknown_fallback(
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    first = _approval(proposal, run_id=11, run_attempt=1)
    with pytest.raises(ValueError, match="github_run_attempt"):
        _approval(proposal, run_id=11, run_attempt=2)
    separate_dispatch = _approval(proposal, run_id=12, run_attempt=1)
    first_body = json.loads(durable_release_body(proposal, first, proposal_repository_commit=_D))
    separate_dispatch_body = durable_release_body(
        proposal, separate_dispatch, proposal_repository_commit=_D
    )
    assert first_body["fallback_terminal"]["protected_source_acquisition"] == (
        "unknown_after_global_consumption"
    )
    assert first_body["github_run_attempt"] == 1
    assert canonical_json_bytes(first_body) != separate_dispatch_body
    durable = _durable(proposal, first)
    assert durable.release_creation_succeeded_in_this_run is True
    with pytest.raises(Item123AuthorizationError, match="does not consume"):
        verify_durable_release(
            proposal,
            first,
            proposal_repository_commit=_D,
            release_payload={
                "id": 123,
                "tag_name": durable.tag_name,
                "target_commitish": _D,
                "name": durable.tag_name,
                "body": durable_release_body(
                    proposal, first, proposal_repository_commit=_D
                ).decode(),
                "draft": False,
                "prerelease": False,
                "immutable": True,
                "assets": [],
            },
            tag_ref_payload={
                "ref": f"refs/tags/{durable.tag_name}",
                "object": {"type": "commit", "sha": _D},
            },
            gh_release_verify_succeeded=True,
            release_creation_succeeded_in_this_run=False,
        )


def test_durable_release_rejects_mutability_assets_body_and_tag_drift(
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    approval = _approval(proposal)
    tag = durable_attempt_tag(proposal.fingerprint)
    base_release: dict[str, Any] = {
        "id": 123,
        "tag_name": tag,
        "target_commitish": _D,
        "name": tag,
        "body": durable_release_body(proposal, approval, proposal_repository_commit=_D).decode(),
        "draft": False,
        "prerelease": False,
        "immutable": True,
        "assets": [],
    }
    base_ref: dict[str, Any] = {
        "ref": f"refs/tags/{tag}",
        "object": {"type": "commit", "sha": _D},
    }
    cases: tuple[tuple[str, object], ...] = (
        ("id", True),
        ("body", "drift"),
        ("immutable", False),
        ("assets", [{"name": "forbidden"}]),
        ("target_commitish", _C),
    )
    for field, value in cases:
        release = deepcopy(base_release)
        release[field] = value
        with pytest.raises(Item123AuthorizationError, match="durable release"):
            verify_durable_release(
                proposal,
                approval,
                proposal_repository_commit=_D,
                release_payload=release,
                tag_ref_payload=base_ref,
                gh_release_verify_succeeded=True,
                release_creation_succeeded_in_this_run=True,
            )
    drifted_ref = deepcopy(base_ref)
    drifted_ref["object"]["sha"] = _C
    with pytest.raises(Item123AuthorizationError, match="does not consume"):
        verify_durable_release(
            proposal,
            approval,
            proposal_repository_commit=_D,
            release_payload=base_release,
            tag_ref_payload=drifted_ref,
            gh_release_verify_succeeded=True,
            release_creation_succeeded_in_this_run=True,
        )
    with pytest.raises(Item123AuthorizationError, match="readback is malformed"):
        verify_durable_release(
            proposal,
            approval,
            proposal_repository_commit=_D,
            release_payload={**base_release, "assets": {}},
            tag_ref_payload=base_ref,
            gh_release_verify_succeeded=True,
            release_creation_succeeded_in_this_run=True,
        )
    with pytest.raises(Item123AuthorizationError, match="proposal commit differs"):
        durable_release_body(proposal, approval, proposal_repository_commit=_C)
    with pytest.raises(Item123AuthorizationError, match="approval or workflow C differs"):
        durable_release_body(
            proposal,
            approval.model_copy(update={"approved_proposal_sha256": "0" * 64}),
            proposal_repository_commit=_D,
        )


def test_local_consumption_and_execution_start_are_atomic_and_nonreplayable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    ledger = tmp_path / "ledger"
    _fake_ledger(monkeypatch, ledger)
    approval = _approval(proposal)
    durable = _durable(proposal, approval)
    with pytest.raises(Item123AuthorizationError, match="ledger root differs"):
        consume_attempt(proposal, approval, durable, ledger_root=tmp_path / "wrong-ledger")
    drifted_durable = durable.model_copy(update={"proposal_sha256": "0" * 64})
    with pytest.raises(Item123AuthorizationError, match="durable consumption differs"):
        consume_attempt(
            proposal,
            approval,
            drifted_durable,
            ledger_root=Path(ITEM12_3_LEDGER_ROOT),
        )
    _, consumption = consume_attempt(
        proposal, approval, durable, ledger_root=Path(ITEM12_3_LEDGER_ROOT)
    )
    capability = _capability(proposal, approval, durable, consumption)

    def claim() -> object:
        try:
            return claim_execution_start(capability)[1]
        except Item123ReplayError:
            return "replay"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))
    starts = [item for item in results if item != "replay"]
    assert len(starts) == 1
    start = cast(authorization.Item123ExecutionStart, starts[0])
    assert start.capability_fingerprint == capability.fingerprint
    assert start.attempt_consumption_fingerprint == consumption.fingerprint
    with pytest.raises(Item123ReplayError):
        claim_execution_start(capability)
    with pytest.raises(Item123AuthorizationError, match="exact capability type"):
        claim_execution_start(cast(Any, SimpleNamespace()))
    with pytest.raises(Item123AuthorizationError, match="exact execution-start type"):
        verify_execution_start(capability, cast(Any, SimpleNamespace()))


def test_execution_start_verifier_rejects_capability_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    ledger = tmp_path / "ledger"
    _fake_ledger(monkeypatch, ledger)
    approval = _approval(proposal)
    durable = _durable(proposal, approval)
    _, consumption = consume_attempt(
        proposal, approval, durable, ledger_root=Path(ITEM12_3_LEDGER_ROOT)
    )
    capability = _capability(proposal, approval, durable, consumption)
    _, start = claim_execution_start(capability)
    real_reader = authorization._read_execution_start
    monkeypatch.setattr(
        authorization,
        "_read_execution_start",
        lambda _: real_reader(ledger / f"{start.attempt_key}.started.json"),
    )
    verify_execution_start(capability, start)
    drifted = start.model_copy(update={"capability_fingerprint": "0" * 64})
    with pytest.raises(Item123AuthorizationError, match="differs"):
        verify_execution_start(capability, drifted)


def test_expired_capability_is_rejected_under_lock_before_started_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    ledger = tmp_path / "ledger"
    _fake_ledger(monkeypatch, ledger)
    approval = _approval(proposal)
    durable = _durable(proposal, approval)
    _, consumption = consume_attempt(
        proposal, approval, durable, ledger_root=Path(ITEM12_3_LEDGER_ROOT)
    )
    capability = _capability(proposal, approval, durable, consumption)
    monkeypatch.setattr(
        authorization,
        "require_proposal_current",
        lambda _: (_ for _ in ()).throw(Item123AuthorizationError("expired")),
    )
    with pytest.raises(Item123AuthorizationError, match="expired"):
        claim_execution_start(capability)
    assert not (ledger / f"{consumption.attempt_key}.started.json").exists()


def test_capability_and_consumption_receipts_round_trip_and_reject_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    ledger = tmp_path / "ledger"
    _fake_ledger(monkeypatch, ledger)
    approval = _approval(proposal)
    durable = _durable(proposal, approval)
    consumption_path, consumption = consume_attempt(
        proposal, approval, durable, ledger_root=Path(ITEM12_3_LEDGER_ROOT)
    )

    durable_path = tmp_path / "durable.json"
    _write_sealed(durable_path, canonical_json_bytes(durable.model_dump(mode="json")))
    runtime_path = tmp_path / "runtime.json"
    _write_sealed(runtime_path, _runtime_payload(proposal))
    assert load_durable_consumption_observation(durable_path) == durable
    assert load_attempt_consumption(consumption_path) == consumption
    assert load_runtime_preparation_payload(runtime_path, proposal=proposal) == _runtime_payload(
        proposal
    )

    capability = _capability(proposal, approval, durable, consumption)
    capability_path = tmp_path / "capability.json"
    _write_sealed(capability_path, canonical_capability_bytes(capability))
    assert load_verified_capability(capability_path) == capability
    with pytest.raises(ValueError, match="broken typed join"):
        VerifiedItem123ExecutionCapability.model_validate(
            {
                **capability.model_dump(mode="python"),
                "approval_fingerprint": "0" * 64,
            }
        )
    mismatched_approval = _approval(proposal, run_id=999)
    with pytest.raises(Item123AuthorizationError, match="mismatched approval evidence"):
        issue_verified_capability(
            proposal,
            mismatched_approval,
            durable,
            consumption,
            _runtime_payload(proposal),
        )

    monkeypatch.setattr(authorization, "_verify_clean_execution_checkout", lambda *args, **kw: None)
    monkeypatch.setattr(authorization, "build_pending_proposal", lambda *args, **kw: proposal)
    monkeypatch.setattr(authorization, "_read_canonical_consumption", lambda _: consumption)
    assert verify_capability_for_execution(capability, repository_root=tmp_path) == consumption
    with pytest.raises(Item123AuthorizationError, match="exact verified capability type"):
        verify_capability_for_execution(cast(Any, SimpleNamespace()), repository_root=tmp_path)

    noncanonical = tmp_path / "capability-noncanonical.json"
    _write_sealed(noncanonical, canonical_capability_bytes(capability) + b"\n")
    with pytest.raises(Item123AuthorizationError, match="canonical JSON"):
        load_verified_capability(noncanonical)
    alias = tmp_path / "capability-link.json"
    alias.symlink_to(capability_path)
    with pytest.raises(Item123AuthorizationError, match="non-hardlinked regular file"):
        load_verified_capability(alias)


def test_real_ledger_permissions_links_and_partial_state_fail_closed(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    ledger.mkdir(mode=0o700)
    _, root_descriptor = authorization._inspect_ledger_root(ledger)
    try:
        lock_descriptor = authorization._lock_ledger(root_descriptor)
        os.close(lock_descriptor)
        with pytest.raises(Item123AuthorizationError, match="payload or name"):
            authorization._write_exclusive_at(root_descriptor, "../escape", b"x", maximum_bytes=10)
    finally:
        os.close(root_descriptor)

    (ledger / ".item12-3.lock").unlink()
    (ledger / "lock-target").write_bytes(b"x")
    (ledger / ".item12-3.lock").symlink_to(ledger / "lock-target")
    _, root_descriptor = authorization._inspect_ledger_root(ledger)
    try:
        with pytest.raises(Item123AuthorizationError, match="cannot acquire"):
            authorization._lock_ledger(root_descriptor)
    finally:
        os.close(root_descriptor)

    ledger.chmod(0o755)
    with pytest.raises(Item123AuthorizationError, match="owned by the runner at 0700"):
        authorization._inspect_ledger_root(ledger)
    ledger.chmod(0o700)
    alias = tmp_path / "ledger-alias"
    alias.symlink_to(ledger, target_is_directory=True)
    with pytest.raises(Item123AuthorizationError, match="non-directory link"):
        authorization._inspect_ledger_root(alias)


def test_runtime_observation_tamper_prevents_capability_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    ledger = tmp_path / "ledger"
    _fake_ledger(monkeypatch, ledger)
    approval = _approval(proposal)
    durable = _durable(proposal, approval)
    _, consumption = consume_attempt(
        proposal, approval, durable, ledger_root=Path(ITEM12_3_LEDGER_ROOT)
    )
    base = json.loads(_runtime_payload(proposal))
    mutations: tuple[tuple[tuple[str, ...], object, str], ...] = (
        (("daemon", "pids_limit_supported"), False, "approved closure"),
        (("daemon", "endpoint"), "tcp://127.0.0.1:2375", "not local and exact"),
        (("archive_file", "inode"), -1, "archive identity"),
        (("archive_file", "mode"), 0o600, "approved closure"),
        (("release", "asset_attestation_verified"), False, "approved closure"),
        (("loaded_image_verified",), False, "approved closure"),
    )
    for path, value, match in mutations:
        payload = deepcopy(base)
        target: dict[str, Any] = payload
        for component in path[:-1]:
            target = cast(dict[str, Any], target[component])
        target[path[-1]] = value
        with pytest.raises((Item123AuthorizationError, ValueError), match=match):
            issue_verified_capability(
                proposal,
                approval,
                durable,
                consumption,
                canonical_json_bytes(payload),
            )
    with pytest.raises(ValueError, match="canonical JSON"):
        issue_verified_capability(
            proposal,
            approval,
            durable,
            consumption,
            _runtime_payload(proposal) + b"\n",
        )


def test_wrapper_runtime_denial_never_claims_or_resolves_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = cast(VerifiedItem123ExecutionCapability, SimpleNamespace())
    monkeypatch.setattr(authorized_runner, "load_verified_capability", lambda _: capability)
    monkeypatch.setattr(
        authorized_runner, "verify_capability_for_execution", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        authorized_runner,
        "_consumption_path",
        lambda _: Path("/source-free/consumed.json"),
    )
    monkeypatch.setattr(
        authorized_runner,
        "_reauthenticate_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("denied")),
    )
    monkeypatch.setattr(
        authorized_runner,
        "claim_execution_start",
        lambda _: pytest.fail("execution was claimed after runtime denial"),
    )
    report = SimpleNamespace(outcome="pre_source_failure")
    monkeypatch.setattr(
        authorized_runner, "record_terminal_report", lambda *args, **kwargs: (Path(), report)
    )
    code, observed = authorized_runner._run(Path("capability"), _ROOT)
    assert code == 1
    assert observed is report


def test_wrapper_happy_path_orders_runtime_claim_source_supervisor_cleanup_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = cast(
        VerifiedItem123ExecutionCapability,
        SimpleNamespace(
            proposal=SimpleNamespace(
                execution_paths=SimpleNamespace(
                    protected_source_path="/fixed/protected/source.h5ad"
                ),
            ),
            runtime_preparation_observation_sha256="1" * 64,
        ),
    )
    events: list[str] = []
    start = cast(authorization.Item123ExecutionStart, SimpleNamespace())
    parent_terminal = SimpleNamespace()
    report = SimpleNamespace(outcome="success")
    monkeypatch.setattr(authorized_runner, "load_verified_capability", lambda _: capability)
    monkeypatch.setattr(
        authorized_runner,
        "verify_capability_for_execution",
        lambda *args, **kwargs: events.append("capability"),
    )
    monkeypatch.setattr(authorized_runner, "_consumption_path", lambda _: Path("consumed.json"))
    monkeypatch.setattr(
        authorized_runner,
        "_reauthenticate_runtime",
        lambda *args: events.append("runtime"),
    )
    monkeypatch.setattr(
        authorized_runner,
        "claim_execution_start",
        lambda _: (Path("started.json"), events.append("claim") or start),
    )
    monkeypatch.setattr(
        authorized_runner,
        "_acquire_source",
        lambda _: (events.append("source") or Path("source.h5ad"), SimpleNamespace()),
    )
    monkeypatch.setattr(
        supervisor,
        "run_contained_training",
        lambda **kwargs: (
            (
                events.append("supervisor") or Path("stage"),
                parent_terminal,
            )
            if kwargs["execution_start"] is start
            else pytest.fail("wrapper dropped execution-start receipt")
        ),
    )
    monkeypatch.setattr(
        authorized_runner,
        "_remove_source",
        lambda *args: events.append("cleanup") or True,
    )
    monkeypatch.setattr(
        authorized_runner,
        "record_contained_terminal_report",
        lambda *args, **kwargs: (events.append("terminal") or Path("terminal"), report),
    )
    code, observed = authorized_runner._run(Path("capability"), _ROOT)
    assert code == 0
    assert observed is report
    assert events == [
        "capability",
        "runtime",
        "claim",
        "source",
        "supervisor",
        "cleanup",
        "terminal",
    ]


def test_wrapper_supervisor_exception_reports_unknown_stage_not_false_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = cast(
        VerifiedItem123ExecutionCapability,
        SimpleNamespace(
            proposal=SimpleNamespace(
                execution_paths=SimpleNamespace(
                    protected_source_path="/fixed/protected/source.h5ad"
                ),
            ),
            runtime_preparation_observation_sha256="1" * 64,
        ),
    )
    monkeypatch.setattr(authorized_runner, "load_verified_capability", lambda _: capability)
    monkeypatch.setattr(
        authorized_runner, "verify_capability_for_execution", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(authorized_runner, "_consumption_path", lambda _: Path("consumed.json"))
    monkeypatch.setattr(authorized_runner, "_reauthenticate_runtime", lambda *args: None)
    monkeypatch.setattr(
        authorized_runner,
        "claim_execution_start",
        lambda _: (Path("started"), cast(authorization.Item123ExecutionStart, SimpleNamespace())),
    )
    monkeypatch.setattr(
        authorized_runner,
        "_acquire_source",
        lambda _: (Path("source.h5ad"), cast(os.stat_result, SimpleNamespace())),
    )
    monkeypatch.setattr(
        supervisor,
        "run_contained_training",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("after-stage-sensitive")),
    )
    monkeypatch.setattr(authorized_runner, "_remove_source", lambda *args: True)
    captured: dict[str, Any] = {}

    def record(*args: object, **kwargs: Any) -> tuple[Path, object]:
        captured.update(kwargs)
        return Path("terminal"), SimpleNamespace(outcome="runtime_failure")

    monkeypatch.setattr(authorized_runner, "record_terminal_report", record)
    code, _ = authorized_runner._run(Path("capability"), _ROOT)
    assert code == 1
    assert captured["stage_disposition"] == "unknown"
    assert captured["stage_semantic_verification"] == "unknown"
    assert captured["stage_preserved_on_runner"] is None


def test_fixed_source_download_uses_no_retry_and_authenticates_before_completion_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    payload = b"source-free-fixture"
    monkeypatch.setattr(authorized_runner, "require_proposal_current", lambda _: None)
    execution_root = tmp_path / "execution"
    execution_root.mkdir(mode=0o700)
    source_parent = execution_root / "protected-source"
    paths = proposal.execution_paths.model_copy(
        update={
            "execution_root": str(execution_root),
            "protected_source_path": str(source_parent / "fixture.h5ad"),
            "source_acquisition_started_marker": str(execution_root / "started.json"),
            "source_acquisition_completed_marker": str(execution_root / "completed.json"),
            "host_source_removed_marker": str(execution_root / "removed.json"),
        }
    )
    source = proposal.protected_source.model_copy(
        update={
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        }
    )
    changed_proposal = proposal.model_copy(
        update={"execution_paths": paths, "protected_source": source}
    )
    capability = cast(
        VerifiedItem123ExecutionCapability,
        SimpleNamespace(
            proposal=changed_proposal,
            attempt_consumption=SimpleNamespace(attempt_key="1" * 64),
            proposal_sha256="2" * 64,
        ),
    )
    commands: list[tuple[str, ...]] = []

    class FakeProcess:
        def __init__(self, command: tuple[str, ...], **_: object) -> None:
            commands.append(command)
            self.stdout = io.BytesIO(payload)

        def wait(self, timeout: float) -> int:
            assert timeout == 10.0
            return 0

        def poll(self) -> int:
            return 0

    monkeypatch.setattr(authorized_runner.subprocess, "Popen", FakeProcess)
    path, identity = authorized_runner._acquire_source(capability)
    assert path.read_bytes() == payload
    assert stat_mode(path) == 0o400
    assert identity.st_size == len(payload)
    command = commands[0]
    assert command[0] == "/usr/bin/curl"
    assert command[command.index("--retry") + 1] == "0"
    assert command[-1] == proposal.protected_source.public_locator_uri


def test_expiry_crossing_before_source_marker_never_starts_curl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = cast(VerifiedItem123ExecutionCapability, SimpleNamespace(proposal=object()))
    monkeypatch.setattr(
        authorized_runner,
        "require_proposal_current",
        lambda _: (_ for _ in ()).throw(Item123AuthorizationError("expired")),
    )
    monkeypatch.setattr(
        authorized_runner,
        "_write_marker",
        lambda *args: pytest.fail("expired capability wrote a source marker"),
    )
    monkeypatch.setattr(
        authorized_runner.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("expired capability started curl"),
    )
    with pytest.raises(Item123AuthorizationError, match="expired"):
        authorized_runner._acquire_source(capability)


def test_run_expiry_crossing_after_claim_defers_to_pre_source_fallback_without_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = cast(
        VerifiedItem123ExecutionCapability,
        SimpleNamespace(
            proposal=SimpleNamespace(
                execution_paths=SimpleNamespace(
                    source_acquisition_started_marker="/never-created/started.json"
                )
            )
        ),
    )
    monkeypatch.setattr(authorized_runner, "load_verified_capability", lambda _: capability)
    monkeypatch.setattr(
        authorized_runner, "verify_capability_for_execution", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(authorized_runner, "_consumption_path", lambda _: Path("consumed.json"))
    monkeypatch.setattr(authorized_runner, "_reauthenticate_runtime", lambda *args: None)
    monkeypatch.setattr(
        authorized_runner,
        "claim_execution_start",
        lambda _: (Path("started"), cast(authorization.Item123ExecutionStart, SimpleNamespace())),
    )
    monkeypatch.setattr(
        authorized_runner,
        "_acquire_source",
        lambda _: (_ for _ in ()).throw(Item123AuthorizationError("expired")),
    )
    monkeypatch.setattr(authorized_runner, "_source_acquisition_may_have_started", lambda _: False)
    monkeypatch.setattr(
        authorized_runner,
        "_remove_source",
        lambda *args: pytest.fail("pre-marker expiry attempted source cleanup"),
    )
    monkeypatch.setattr(
        authorized_runner,
        "record_terminal_report",
        lambda *args, **kwargs: pytest.fail(
            "post-claim pre-marker denial installed a competing no-source terminal"
        ),
    )
    with pytest.raises(Item123AuthorizationError, match="did not start after the exclusive"):
        authorized_runner._run(Path("capability"), _ROOT)


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _proposal_dispatch_repository(
    tmp_path: Path,
    *,
    extra_dispatch_file: bool = False,
    executable_proposal: bool = False,
    proposal_payload_suffix: bytes = b"",
) -> tuple[Path, authorization.Item123ExecutionProposal, str]:
    tmp_path.mkdir(parents=True)
    repository = tmp_path / "dispatch"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "source-free@example.invalid")
    _git(repository, "config", "user.name", "Source Free Test")
    (repository / "README").write_text("execution C\n", encoding="utf-8")
    _git(repository, "add", "README")
    _git(repository, "commit", "--quiet", "-m", "execution C")
    execution_commit = _git(repository, "rev-parse", "HEAD")
    proposal = build_pending_proposal(
        _ROOT,
        execution_repository_commit=execution_commit,
        not_before_utc="2026-08-11T00:00:00Z",
        expires_at_utc="2026-08-12T00:00:00Z",
    )
    proposal_path = repository / authorization.ITEM12_3_PROPOSAL_RELATIVE_PATH
    proposal_path.parent.mkdir(parents=True)
    proposal_path.write_bytes(canonical_proposal_bytes(proposal) + proposal_payload_suffix)
    if executable_proposal:
        proposal_path.chmod(0o755)
    _git(repository, "add", authorization.ITEM12_3_PROPOSAL_RELATIVE_PATH)
    if extra_dispatch_file:
        (repository / "unexpected.txt").write_text("drift\n", encoding="utf-8")
        _git(repository, "add", "unexpected.txt")
    _git(repository, "commit", "--quiet", "-m", "proposal D")
    proposal_commit = _git(repository, "rev-parse", "HEAD")
    _git(repository, "checkout", "--quiet", "--detach", execution_commit)
    return repository, proposal, proposal_commit


def test_proposal_only_dispatch_topology_accepts_exact_addition_and_rejects_drift(
    tmp_path: Path,
) -> None:
    repository, proposal, dispatch = _proposal_dispatch_repository(tmp_path / "valid")
    authorization._verify_proposal_only_commit(
        repository, proposal, proposal_repository_commit=dispatch
    )
    (repository / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(Item123AuthorizationError, match="repository drift"):
        authorization._verify_proposal_only_commit(
            repository, proposal, proposal_repository_commit=dispatch
        )

    extra_repository, extra_proposal, extra_dispatch = _proposal_dispatch_repository(
        tmp_path / "extra", extra_dispatch_file=True
    )
    with pytest.raises(Item123AuthorizationError, match="more than the one proposal"):
        authorization._verify_proposal_only_commit(
            extra_repository,
            extra_proposal,
            proposal_repository_commit=extra_dispatch,
        )

    mode_repository, mode_proposal, mode_dispatch = _proposal_dispatch_repository(
        tmp_path / "executable", executable_proposal=True
    )
    with pytest.raises(Item123AuthorizationError, match="exact 100644 blob"):
        authorization._verify_proposal_only_commit(
            mode_repository,
            mode_proposal,
            proposal_repository_commit=mode_dispatch,
        )

    bytes_repository, bytes_proposal, bytes_dispatch = _proposal_dispatch_repository(
        tmp_path / "bytes", proposal_payload_suffix=b"\n"
    )
    with pytest.raises(Item123AuthorizationError, match="differs from approved bytes"):
        authorization._verify_proposal_only_commit(
            bytes_repository,
            bytes_proposal,
            proposal_repository_commit=bytes_dispatch,
        )


def test_clean_execution_commit_gate_and_source_free_proposal_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    repository = tmp_path / "execution"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "source-free@example.invalid")
    _git(repository, "config", "user.name", "Source Free Test")
    (repository / "bound.txt").write_text("bound\n", encoding="utf-8")
    _git(repository, "add", "bound.txt")
    _git(repository, "commit", "--quiet", "-m", "execution C")
    commit = _git(repository, "rev-parse", "HEAD")
    authorization._verify_clean_execution_checkout(repository, expected_commit=commit)
    with pytest.raises(Item123AuthorizationError, match="HEAD differs"):
        authorization._verify_clean_execution_checkout(repository, expected_commit="0" * 40)
    (repository / "drift.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(Item123AuthorizationError, match="tracked or untracked drift"):
        authorization._verify_clean_execution_checkout(repository, expected_commit=commit)

    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_bytes(canonical_proposal_bytes(proposal))
    approval = _approval(proposal)
    monkeypatch.setattr(authorization, "require_proposal_current", lambda *args, **kwargs: None)
    monkeypatch.setattr(authorization, "_verify_proposal_only_commit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        authorization, "_verify_clean_execution_checkout", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(authorization, "build_pending_proposal", lambda *args, **kwargs: proposal)
    observed, observed_approval = verify_proposal_against_repository(
        proposal_path,
        proposal.fingerprint,
        tmp_path,
        dispatch_actor=approval.dispatch_actor,
        triggering_actor=approval.triggering_actor,
        github_run_id=approval.github_run_id,
        github_run_attempt=approval.github_run_attempt,
        workflow_dispatch_ref=approval.workflow_dispatch_ref,
        workflow_repository_commit=approval.workflow_repository_commit,
        proposal_repository_commit=approval.proposal_repository_commit,
        current_time_utc=datetime(2026, 8, 11, 12, tzinfo=UTC),
    )
    assert observed == proposal
    assert observed_approval == approval
    monkeypatch.setattr(
        authorization,
        "build_pending_proposal",
        lambda *args, **kwargs: proposal.model_copy(
            update={"execution_repository_commit": "0" * 40}
        ),
    )
    with pytest.raises(Item123AuthorizationError, match="current exact public bindings"):
        verify_proposal_against_repository(
            proposal_path,
            proposal.fingerprint,
            tmp_path,
            dispatch_actor=approval.dispatch_actor,
            triggering_actor=approval.triggering_actor,
            github_run_id=approval.github_run_id,
            github_run_attempt=approval.github_run_attempt,
            workflow_dispatch_ref=approval.workflow_dispatch_ref,
            workflow_repository_commit=approval.workflow_repository_commit,
            proposal_repository_commit=approval.proposal_repository_commit,
            current_time_utc=datetime(2026, 8, 11, 12, tzinfo=UTC),
        )


def test_terminal_digest_names_bind_parent_and_staged_success_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    success = SimpleNamespace(model_dump=lambda mode: {"success": True})
    execution = SimpleNamespace(outcome="success", parent_wall_clock_elapsed_seconds=1.0)
    worker = SimpleNamespace(
        source_pre_sha256="1" * 64,
        source_post_sha256="1" * 64,
        source_matches_expected=True,
    )
    Terminal = type(
        "ContainedTrainingTerminalObservation",
        (),
        {"__module__": "cellstate.training.execution"},
    )
    terminal = Terminal()
    terminal.protected_source_acquired_before_supervisor = True
    terminal.terminal_status = "success"
    terminal.failure_code = "none"
    terminal.execution_observation = execution
    terminal.worker_terminal_report = worker
    terminal.aggregate_container_limits_enforced = True
    terminal.container_cleanup_disposition = "proved_removed"
    terminal.snapshot_volume_cleanup_disposition = "proved_removed"
    terminal.worker_report_status = "verified"
    terminal.stage_disposition = "sealed"
    terminal.semantic_stage_sha256 = "2" * 64
    terminal.canonical_publication_unchanged = True
    terminal.staged_tree_sha256 = "3" * 64
    terminal.success_observation = success
    terminal.model_dump = lambda mode: {"parent": True}
    captured: dict[str, Any] = {}

    def record(*args: object, **kwargs: Any) -> tuple[Path, object]:
        captured.update(kwargs)
        return Path("terminal"), SimpleNamespace()

    monkeypatch.setattr(authorization, "record_terminal_report", record)
    record_contained_terminal_report(
        Path("consumed"),
        terminal,
        protected_source_acquired=True,
        runtime_preparation_observation_sha256="4" * 64,
        protected_source_cleanup_disposition="removed",
    )
    assert (
        captured["parent_terminal_observation_sha256"]
        == hashlib.sha256(canonical_json_bytes({"parent": True})).hexdigest()
    )
    assert (
        captured["staged_success_observation_sha256"]
        == hashlib.sha256(canonical_json_bytes({"success": True})).hexdigest()
    )
    assert (
        captured["parent_terminal_observation_sha256"]
        != captured["staged_success_observation_sha256"]
    )
    terminal.container_cleanup_disposition = "unproved"
    terminal.snapshot_volume_cleanup_disposition = "unproved"
    captured.clear()
    record_contained_terminal_report(
        Path("consumed"),
        terminal,
        protected_source_acquired=True,
        runtime_preparation_observation_sha256="4" * 64,
        protected_source_cleanup_disposition="removed",
    )
    assert captured["outcome"] == "runtime_failure"
    assert captured["protected_source_cleanup_disposition"] == "unknown"
    assert captured["host_source_cleanup_disposition"] == "removed"

    terminal.container_cleanup_disposition = "proved_removed"
    terminal.snapshot_volume_cleanup_disposition = "proved_removed"
    worker.source_matches_expected = False
    worker.source_post_sha256 = None
    captured.clear()
    record_contained_terminal_report(
        Path("consumed"),
        terminal,
        protected_source_acquired=True,
        runtime_preparation_observation_sha256="4" * 64,
        protected_source_cleanup_disposition="removed",
    )
    assert captured["source_match_disposition"] == "pre_match_post_failed"
    worker.source_post_sha256 = "9" * 64
    captured.clear()
    record_contained_terminal_report(
        Path("consumed"),
        terminal,
        protected_source_acquired=True,
        runtime_preparation_observation_sha256="4" * 64,
        protected_source_cleanup_disposition="removed",
    )
    assert captured["source_match_disposition"] == "mismatch"
    terminal.worker_terminal_report = None
    terminal.semantic_stage_sha256 = None
    terminal.terminal_status = "stage_rejected"
    terminal.failure_code = "stage_semantic_verification_failed"
    terminal.success_observation = None
    captured.clear()
    record_contained_terminal_report(
        Path("consumed"),
        terminal,
        protected_source_acquired=True,
        runtime_preparation_observation_sha256="4" * 64,
        protected_source_cleanup_disposition="removed",
    )
    assert captured["source_match_disposition"] == "acquired_unverified"
    assert captured["stage_semantic_verification"] == "failed"

    with pytest.raises(Item123AuthorizationError, match="exact parent type"):
        record_contained_terminal_report(
            Path("consumed"),
            SimpleNamespace(),
            protected_source_acquired=True,
            runtime_preparation_observation_sha256="4" * 64,
            protected_source_cleanup_disposition="removed",
        )
    terminal.terminal_status = "outside-vocabulary"
    with pytest.raises(Item123AuthorizationError, match="bounded vocabulary"):
        record_contained_terminal_report(
            Path("consumed"),
            terminal,
            protected_source_acquired=True,
            runtime_preparation_observation_sha256="4" * 64,
            protected_source_cleanup_disposition="removed",
        )


def test_terminal_ledger_records_pre_source_partial_and_success_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    ledger = tmp_path / "ledger"
    _fake_ledger(monkeypatch, ledger)

    pre_approval = _approval(proposal, run_id=31)
    pre_durable = _durable(proposal, pre_approval)
    pre_consumption_path, pre_consumption = consume_attempt(
        proposal, pre_approval, pre_durable, ledger_root=Path(ITEM12_3_LEDGER_ROOT)
    )
    pre_capability = _capability(proposal, pre_approval, pre_durable, pre_consumption)
    pre_path, pre_report = record_terminal_report(
        pre_consumption_path,
        outcome="pre_source_failure",
        reason="runtime_distribution_failed",
        protected_source_acquired=False,
    )
    assert pre_report.outcome == "pre_source_failure"
    assert pre_report.protected_source_cleanup_disposition == "not_applicable"
    assert stat_mode(pre_path) == 0o400
    with pytest.raises(Item123ReplayError, match="already has terminal evidence"):
        record_terminal_report(
            pre_consumption_path,
            outcome="pre_source_failure",
            reason="runtime_distribution_failed",
            protected_source_acquired=False,
        )
    with pytest.raises(Item123ReplayError, match="already started or terminated"):
        claim_execution_start(pre_capability)

    partial_approval = _approval(proposal, run_id=32)
    partial_durable = _durable(proposal, partial_approval)
    partial_consumption_path, partial_consumption = consume_attempt(
        proposal,
        partial_approval,
        partial_durable,
        ledger_root=Path(ITEM12_3_LEDGER_ROOT),
    )
    partial_capability = _capability(
        proposal, partial_approval, partial_durable, partial_consumption
    )
    claim_execution_start(partial_capability)
    with pytest.raises(Item123ReplayError, match="after execution started"):
        record_terminal_report(
            partial_consumption_path,
            outcome="pre_source_failure",
            reason="runtime_distribution_failed",
            protected_source_acquired=False,
        )
    assert not (ledger / f"{partial_consumption.attempt_key}.terminal.json").exists()
    _, partial_report = record_terminal_report(
        partial_consumption_path,
        outcome="runtime_failure",
        reason="protected_source_acquisition_failed",
        protected_source_acquired=True,
        protected_source_acquisition="started_incomplete",
        protected_source_cleanup_disposition="removed",
        host_source_cleanup_disposition="removed",
        runtime_preparation_observation_sha256=(
            partial_capability.runtime_preparation_observation_sha256
        ),
        source_match_disposition="acquired_unverified",
    )
    assert partial_report.protected_source_cleanup_disposition == "removed"
    assert partial_report.container_cleanup_disposition == "not_started"

    success_approval = _approval(proposal, run_id=33)
    success_durable = _durable(proposal, success_approval)
    success_consumption_path, success_consumption = consume_attempt(
        proposal,
        success_approval,
        success_durable,
        ledger_root=Path(ITEM12_3_LEDGER_ROOT),
    )
    success_capability = _capability(
        proposal, success_approval, success_durable, success_consumption
    )
    claim_execution_start(success_capability)
    _, success_report = record_terminal_report(
        success_consumption_path,
        outcome="success",
        reason="stage_sealed",
        protected_source_acquired=True,
        protected_source_acquisition="completed",
        protected_source_cleanup_disposition="removed",
        host_source_cleanup_disposition="removed",
        container_cleanup_disposition="proved_removed",
        snapshot_volume_cleanup_disposition="proved_removed",
        contained_terminal_status="success",
        contained_failure_code="none",
        execution_outcome="success",
        parent_wall_clock_elapsed_seconds=1.0,
        aggregate_container_limits_enforced=True,
        runtime_preparation_observation_sha256=(
            success_capability.runtime_preparation_observation_sha256
        ),
        source_match_disposition="pre_and_post_match",
        worker_report_status="verified",
        stage_disposition="sealed",
        stage_semantic_verification="verified",
        canonical_publication_unchanged=True,
        parent_terminal_observation_sha256="1" * 64,
        staged_success_observation_sha256="2" * 64,
        staged_tree_sha256="3" * 64,
        stage_preserved_on_runner=True,
    )
    assert success_report.outcome == "success"
    assert success_report.fingerprint == authorization.canonical_fingerprint(
        success_report.model_dump(mode="json")
    )


def test_terminal_schema_rejects_false_source_cleanup_and_stage_claims(
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    approval = _approval(proposal)
    consumption = authorization.Item123AttemptConsumption(
        attempt_key="1" * 64,
        proposal_sha256=proposal.fingerprint,
        proposal_fingerprint=proposal.fingerprint,
        approval_fingerprint=approval.fingerprint,
        durable_consumption_fingerprint="2" * 64,
        execution_repository_commit=proposal.execution_repository_commit,
        execution_id=proposal.execution_paths.execution_id,
        protected_source_acquired_at_consumption=False,
        attempt_count=1,
        no_retry=True,
    )
    base = {
        "attempt_key": consumption.attempt_key,
        "attempt_consumption_fingerprint": consumption.fingerprint,
        "proposal_sha256": proposal.fingerprint,
        "execution_id": proposal.execution_paths.execution_id,
        "outcome": "pre_source_failure",
        "reason": "runtime_distribution_failed",
        "contained_terminal_status": "not_started",
        "contained_failure_code": "not_applicable",
        "execution_outcome": "not_started",
        "parent_wall_clock_elapsed_seconds": None,
        "aggregate_container_limits_enforced": False,
        "runtime_preparation_observation_sha256": None,
        "protected_source_acquired": False,
        "protected_source_acquisition": "not_started",
        "protected_source_cleanup_disposition": "not_applicable",
        "host_source_cleanup_disposition": "not_applicable",
        "container_cleanup_disposition": "not_started",
        "snapshot_volume_cleanup_disposition": "not_started",
        "source_match_disposition": "not_acquired",
        "worker_report_status": "not_started",
        "stage_disposition": "not_created",
        "stage_semantic_verification": "not_attempted",
        "canonical_publication_unchanged": None,
        "parent_terminal_observation_sha256": None,
        "staged_success_observation_sha256": None,
        "staged_tree_sha256": None,
        "stage_preserved_on_runner": False,
        "canonical_publication_performed": False,
        "nonterminal_artifact_uploaded": False,
        "terminal_report_upload_permitted": True,
        "uploaded_payload_scope": "terminal-report-only",
        "retry_permitted": False,
    }
    mutations: tuple[tuple[str, object, str], ...] = (
        ("reason", "stage_sealed", "failed terminal report"),
        ("protected_source_acquired", True, "pre-source terminal failure"),
        ("protected_source_cleanup_disposition", "removed", "cleanup disposition"),
        ("host_source_cleanup_disposition", "removed", "invents cleanup evidence"),
        ("source_match_disposition", "acquired_unverified", "identity disposition"),
        ("stage_disposition", "sealed", "contained execution evidence"),
    )
    for field, value, match in mutations:
        with pytest.raises(ValueError, match=match):
            authorization.Item123TerminalReport.model_validate({**base, field: value})

    with pytest.raises(ValueError, match="lacks sealed-stage evidence"):
        authorization.Item123TerminalReport.model_validate(
            {**base, "outcome": "success", "reason": "stage_sealed"}
        )
    with pytest.raises(ValueError, match="started flag"):
        authorization.Item123TerminalReport.model_validate(
            {**base, "protected_source_acquisition": "started_incomplete"}
        )
    acquired = {
        **base,
        "outcome": "runtime_failure",
        "reason": "protected_source_acquisition_failed",
        "protected_source_acquired": True,
        "protected_source_acquisition": "started_incomplete",
        "protected_source_cleanup_disposition": "unknown",
        "host_source_cleanup_disposition": "removed",
        "container_cleanup_disposition": "unproved",
        "snapshot_volume_cleanup_disposition": "unproved",
        "source_match_disposition": "acquired_unverified",
    }
    with pytest.raises(ValueError, match="without verified runtime preparation"):
        authorization.Item123TerminalReport.model_validate(acquired)
    with pytest.raises(ValueError, match="complete cleanup proof"):
        authorization.Item123TerminalReport.model_validate(
            {
                **acquired,
                "protected_source_cleanup_disposition": "removed",
                "runtime_preparation_observation_sha256": "4" * 64,
            }
        )
    with pytest.raises(ValueError, match="failed host cleanup"):
        authorization.Item123TerminalReport.model_validate(
            {
                **acquired,
                "host_source_cleanup_disposition": "failed",
                "runtime_preparation_observation_sha256": "4" * 64,
            }
        )
    success = {
        **acquired,
        "outcome": "success",
        "reason": "stage_sealed",
        "protected_source_acquisition": "completed",
        "protected_source_cleanup_disposition": "removed",
        "container_cleanup_disposition": "proved_removed",
        "snapshot_volume_cleanup_disposition": "proved_removed",
        "contained_terminal_status": "success",
        "contained_failure_code": "none",
        "execution_outcome": "success",
        "parent_wall_clock_elapsed_seconds": 1.0,
        "aggregate_container_limits_enforced": True,
        "runtime_preparation_observation_sha256": "4" * 64,
        "source_match_disposition": "pre_and_post_match",
        "worker_report_status": "missing",
        "stage_disposition": "sealed",
        "stage_semantic_verification": "verified",
        "canonical_publication_unchanged": True,
        "parent_terminal_observation_sha256": "5" * 64,
        "staged_success_observation_sha256": "6" * 64,
        "staged_tree_sha256": "7" * 64,
        "stage_preserved_on_runner": True,
    }
    with pytest.raises(ValueError, match="contradicts contained evidence"):
        authorization.Item123TerminalReport.model_validate(success)


def test_execution_start_and_no_source_terminal_form_one_atomic_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    ledger = tmp_path / "ledger"
    _fake_ledger(monkeypatch, ledger)
    approval = _approval(proposal, run_id=34)
    durable = _durable(proposal, approval)
    consumption_path, consumption = consume_attempt(
        proposal, approval, durable, ledger_root=Path(ITEM12_3_LEDGER_ROOT)
    )
    capability = _capability(proposal, approval, durable, consumption)
    barrier = Barrier(2)

    def claim() -> str:
        barrier.wait()
        try:
            claim_execution_start(capability)
        except Item123ReplayError:
            return "start-denied"
        return "started"

    def terminalize() -> str:
        barrier.wait()
        try:
            record_terminal_report(
                consumption_path,
                outcome="pre_source_failure",
                reason="runtime_distribution_failed",
                protected_source_acquired=False,
            )
        except Item123ReplayError:
            return "terminal-denied"
        return "terminal"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(claim), pool.submit(terminalize))
        results = {future.result() for future in futures}
    assert results in ({"started", "terminal-denied"}, {"start-denied", "terminal"})
    started = (ledger / f"{consumption.attempt_key}.started.json").exists()
    terminal = (ledger / f"{consumption.attempt_key}.terminal.json").exists()
    assert started is not terminal


def test_workflow_is_fixed_ref_post_consumption_runtime_and_terminal_only() -> None:
    workflow = (_ROOT / ".github/workflows/item12-3-sciplex3-v5.yml").read_text()
    assert 'test "${EVENT_REF}" = "refs/heads/main"' in workflow
    assert "proposal_commit:" in workflow
    assert 'test "${EVENT_WORKFLOW_REF}" = \\' in workflow
    assert (
        '"logannye/cellstate/.github/workflows/item12-3-sciplex3-v5.yml@refs/heads/main"'
        in workflow
    )
    assert 'test "${EVENT_WORKFLOW_SHA}" = "${EVENT_SHA}"' in workflow
    assert 'test "${RUN_ATTEMPT}" = "1"' in workflow
    assert '--workflow-dispatch-ref "${EVENT_REF}"' in workflow
    assert '--workflow-repository-commit "${WORKFLOW_COMMIT}"' in workflow
    assert '--proposal-repository-commit "${PROPOSAL_COMMIT}"' in workflow
    assert '--durable-observation-output "${durable_observation}"' in workflow
    assert '--capability-output "${capability}"' in workflow
    assert workflow.index("verify-release-and-consume") < workflow.index("prepare-runtime")
    assert workflow.index("prepare-runtime") < workflow.index("issue-capability")
    assert workflow.index("issue-capability") < workflow.index(
        "run_item12_3_authorized_execution.py"
    )
    assert "if: always()" in workflow
    assert "path: ${{ runner.temp }}/item12-3-terminal-report.json" in workflow
    assert "include-hidden-files: false" in workflow
    assert "--host-source-cleanup-disposition" in workflow
    assert "--protected-source-cleanup-disposition" not in workflow
    assert "**" not in workflow


def test_workflow_authenticates_data_only_d_before_setup_or_repository_execution() -> None:
    workflow = (_ROOT / ".github/workflows/item12-3-sciplex3-v5.yml").read_text()
    initial = workflow.index(
        "- name: Reject the wrong trusted workflow identity, actor, rerun, or input shape"
    )
    checkout = workflow.index(
        "- name: Check out trusted workflow and execution commit C without retaining credentials"
    )
    gate = workflow.index(
        "- name: Authenticate proposal commit D as inert data before repository execution"
    )
    setup = workflow.index("- name: Set up exact public verification runtime")
    install = workflow.index("- name: Install locked source-free verifier dependencies")
    typed = workflow.index("- name: Reverify proposal topology and bindings with trusted C tooling")
    assert initial < checkout < gate < setup < install < typed
    gate_body = workflow[gate:setup]
    assert "fetch-depth: 0" in workflow[:gate]
    assert "ref: ${{ github.sha }}" in workflow[:gate]
    assert "persist-credentials: false" in workflow[:gate]
    assert "git_safe=(/usr/bin/git" in gate_body
    assert "diff-tree --no-renames" in gate_body
    assert 'tree_prefix="100644 blob "' in gate_body
    assert 'cat-file blob "${object_spec}"' in gate_body
    assert "/usr/bin/sha256sum" in gate_body
    assert "uv " not in gate_body
    assert "scripts/" not in gate_body
    assert "python" not in gate_body.lower()
    assert '"${ATTEMPT_TAG}" "${PROPOSAL_COMMIT_INPUT}"' in workflow
    action_lines = [line.strip() for line in workflow.splitlines() if "uses:" in line]
    assert len(action_lines) == 4
    for line in action_lines:
        reference = line.split("@", maxsplit=1)[1].split(maxsplit=1)[0]
        assert len(reference) == 40
        assert all(character in "0123456789abcdef" for character in reference)


def test_workflow_fallback_selection_preserves_partial_acquisition_truth() -> None:
    workflow = (_ROOT / ".github/workflows/item12-3-sciplex3-v5.yml").read_text()
    assert "render_fallback started_incomplete removed" in workflow
    assert "render_fallback completed removed" in workflow
    finalizer = workflow[
        workflow.index(
            "- name: Finalize exactly one bounded sanitized terminal report"
        ) : workflow.index("- name: Upload only sanitized terminal evidence")
    ]
    assert 'selected="${FALLBACK_STARTED_REMOVED}"' in finalizer
    assert 'selected="${FALLBACK_COMPLETED_REMOVED}"' in finalizer
    assert finalizer.index('selected="${FALLBACK_STARTED_REMOVED}"') < finalizer.index(
        'selected="${FALLBACK_COMPLETED}"'
    )


def test_workflow_downloads_private_runtime_asset_with_bounded_bearer_auth() -> None:
    workflow = (_ROOT / ".github/workflows/item12-3-sciplex3-v5.yml").read_text()
    runtime_step = workflow[
        workflow.index(
            "- name: Download, authenticate, load, and reauthenticate exact runtime archive"
        ) : workflow.index("- name: Issue exact consumed runtime-bound capability")
    ]
    assert "GH_TOKEN: ${{ github.token }}" in runtime_step
    assert 'auth="Authorization: Bearer ${GH_TOKEN}"' in runtime_step
    assert 'accept_asset="Accept: application/octet-stream"' in runtime_step
    assert '"${api}/releases/tags/${release_tag}"' in runtime_step
    assert "https://api.github.com/repos/logannye/cellstate/releases/assets/" in runtime_step
    assert 'test "${redirect_status}" = "302"' in runtime_step
    assert 'parsed.hostname != "release-assets.githubusercontent.com"' in runtime_step
    assert 'parsed.path.startswith("/github-production-release-asset/")' in runtime_step
    redirect_request = runtime_step[
        runtime_step.index('redirect_status="$(') : runtime_step.index(
            'test "${redirect_status}" = "302"'
        )
    ]
    assert '--header "${accept_asset}" --header "${auth}" --header "${version}"' in (
        redirect_request
    )
    assert "--location" not in redirect_request
    download_request = runtime_step[
        runtime_step.index('download_status="$(') : runtime_step.index(
            'test "${download_status}" = "200"'
        )
    ]
    assert '"${signed_asset_url}"' in download_request
    assert "${auth}" not in download_request
    assert "--header" not in download_request
    assert "--location" not in download_request
    assert runtime_step.count("--retry 0") == 3
    assert "--location" not in runtime_step
    assert "releases/download/" not in runtime_step
    assert 'echo "${GH_TOKEN}"' not in runtime_step
    assert "set -x" not in runtime_step


def test_workflow_authorization_command_argv_matches_cli_parser() -> None:
    common = [
        "--proposal",
        "proposal.json",
        "--approved-proposal-sha256",
        "1" * 64,
        "--repository-root",
        ".",
        "--dispatch-actor",
        "logannye",
        "--triggering-actor",
        "logannye",
        "--github-run-id",
        "10",
        "--github-run-attempt",
        "1",
        "--workflow-dispatch-ref",
        ITEM12_3_WORKFLOW_DISPATCH_REF,
        "--workflow-repository-commit",
        _C,
        "--proposal-repository-commit",
        _D,
    ]
    parser = authorization_cli._parser()
    assert parser.parse_args(["verify-proposal", *common]).command == "verify-proposal"
    assert (
        parser.parse_args(["render-release-body", *common, "--output", "body.json"]).command
        == "render-release-body"
    )
    assert (
        parser.parse_args(
            [
                "verify-release-and-consume",
                *common,
                "--release-json",
                "release.json",
                "--tag-ref-json",
                "tag.json",
                "--gh-release-verified",
                "--release-created-in-this-run",
                "--durable-observation-output",
                "durable.json",
            ]
        ).command
        == "verify-release-and-consume"
    )
    assert (
        parser.parse_args(
            [
                "issue-capability",
                *common,
                "--durable-observation",
                "durable.json",
                "--consumption",
                "consumed.json",
                "--runtime-preparation-observation",
                "runtime.json",
                "--capability-output",
                "capability.json",
            ]
        ).command
        == "issue-capability"
    )
    assert (
        parser.parse_args(
            [
                "render-fallback-terminal",
                *common,
                "--protected-source-acquisition",
                "completed",
                "--host-source-cleanup-disposition",
                "removed",
                "--output",
                "fallback.json",
            ]
        ).command
        == "render-fallback-terminal"
    )


def test_fallback_unknown_state_is_conservative(
    proposal: authorization.Item123ExecutionProposal,
) -> None:
    report = build_durable_fallback_terminal(
        proposal,
        _approval(proposal),
        protected_source_acquisition="unknown_after_global_consumption",
        host_source_cleanup_disposition="unknown",
    )
    assert report.protected_source_acquired is None
    assert report.source_match_disposition == "unknown"
    assert len(canonical_json_bytes(report.model_dump(mode="json"))) <= 4096
    pre_source = build_durable_fallback_terminal(
        proposal,
        _approval(proposal),
        protected_source_acquisition="not_started",
        host_source_cleanup_disposition="not_applicable",
    )
    assert pre_source.protected_source_cleanup_disposition == "not_applicable"
    completed_host_removed = build_durable_fallback_terminal(
        proposal,
        _approval(proposal),
        protected_source_acquisition="completed",
        host_source_cleanup_disposition="removed",
    )
    assert completed_host_removed.host_source_cleanup_disposition == "removed"
    assert completed_host_removed.protected_source_cleanup_disposition == "unknown"
    assert completed_host_removed.container_cleanup_disposition == "unknown"
    partial_host_removed = build_durable_fallback_terminal(
        proposal,
        _approval(proposal),
        protected_source_acquisition="started_incomplete",
        host_source_cleanup_disposition="removed",
    )
    assert partial_host_removed.protected_source_acquired is True
    assert partial_host_removed.protected_source_acquisition == "started_incomplete"
    assert partial_host_removed.host_source_cleanup_disposition == "removed"
    assert partial_host_removed.protected_source_cleanup_disposition == "unknown"
    with pytest.raises(ValueError, match="specific claim"):
        authorization.Item123DurableFallbackTerminalReport.model_validate(
            {
                **report.model_dump(mode="json"),
                "host_source_cleanup_disposition": "removed",
            }
        )
    pre_payload = pre_source.model_dump(mode="json")
    with pytest.raises(ValueError, match="acquisition state"):
        authorization.Item123DurableFallbackTerminalReport.model_validate(
            {**pre_payload, "protected_source_acquired": True}
        )
    with pytest.raises(ValueError, match="identity state"):
        authorization.Item123DurableFallbackTerminalReport.model_validate(
            {**pre_payload, "source_match_disposition": "acquired_unverified"}
        )
    with pytest.raises(ValueError, match="cleanup state"):
        authorization.Item123DurableFallbackTerminalReport.model_validate(
            {**pre_payload, "protected_source_cleanup_disposition": "unknown"}
        )
    with pytest.raises(ValueError, match="invents cleanup activity"):
        authorization.Item123DurableFallbackTerminalReport.model_validate(
            {**pre_payload, "host_source_cleanup_disposition": "removed"}
        )
    with pytest.raises(ValueError, match="complete contained cleanup"):
        authorization.Item123DurableFallbackTerminalReport.model_validate(
            {
                **completed_host_removed.model_dump(mode="json"),
                "container_cleanup_disposition": "not_started",
            }
        )
    assert len(completed_host_removed.fingerprint) == 64
