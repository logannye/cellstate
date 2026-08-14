#!/usr/bin/env python3
"""Build and verify the source-free, one-use Item 12.3 authorization boundary."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

from cellstate.domain.common import canonical_json_bytes
from cellstate.training.item12_3_authorization import (
    ITEM12_3_MAX_PROPOSAL_BYTES,
    Item123AuthorizationError,
    Item123DurableFallbackTerminalReport,
    Item123ExecutionProposal,
    Item123TerminalReport,
    build_durable_fallback_terminal,
    build_pending_proposal,
    canonical_capability_bytes,
    canonical_proposal_bytes,
    consume_attempt,
    durable_attempt_tag,
    durable_release_body,
    issue_verified_capability,
    load_attempt_consumption,
    load_durable_consumption_observation,
    load_runtime_preparation_payload,
    load_verified_capability,
    record_terminal_report,
    verify_capability_for_execution,
    verify_durable_release,
    verify_proposal_against_repository,
)


def _write_exclusive(path: Path, payload: bytes, *, maximum_bytes: int) -> None:
    if not payload or len(payload) > maximum_bytes:
        raise Item123AuthorizationError("requested authorization output violates its size bound")
    destination = Path(os.path.abspath(path))  # noqa: PTH100 - preserve no-follow semantics
    try:
        parent_state = destination.parent.lstat()
    except OSError as error:
        raise Item123AuthorizationError("authorization output parent is unavailable") from error
    if not stat.S_ISDIR(parent_state.st_mode) or stat.S_ISLNK(parent_state.st_mode):
        raise Item123AuthorizationError("authorization output parent must be one real directory")
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    parent_descriptor = -1
    try:
        parent_descriptor = os.open(destination.parent, parent_flags)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(destination.name, flags, 0o400, dir_fd=parent_descriptor)
        os.fchmod(descriptor, 0o400)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise Item123AuthorizationError("authorization output write was incomplete")
            offset += written
        os.fsync(descriptor)
        os.fsync(parent_descriptor)
    except FileExistsError as error:
        raise Item123AuthorizationError("authorization output already exists") from error
    except OSError as error:
        raise Item123AuthorizationError("cannot seal authorization output") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _read_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        state = path.lstat()
        if not stat.S_ISREG(state.st_mode) or stat.S_ISLNK(state.st_mode):
            raise Item123AuthorizationError(f"{name} must be one regular file")
        if state.st_size < 2 or state.st_size > 1024 * 1024:
            raise Item123AuthorizationError(f"{name} violates its size bound")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            payload = b""
            while len(payload) < opened.st_size:
                chunk = os.read(descriptor, opened.st_size - len(payload))
                if not chunk:
                    raise Item123AuthorizationError(f"{name} ended unexpectedly")
                payload += chunk
            if os.read(descriptor, 1):
                raise Item123AuthorizationError(f"{name} grew while being read")
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise Item123AuthorizationError(f"cannot read {name}") from error
    if (opened.st_dev, opened.st_ino, opened.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise Item123AuthorizationError(f"{name} changed while being read")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Item123AuthorizationError(f"{name} is not valid JSON") from error
    if type(value) is not dict:
        raise Item123AuthorizationError(f"{name} must be one JSON object")
    return value


def _verified(
    args: argparse.Namespace,
) -> tuple[Item123ExecutionProposal, Any]:
    return verify_proposal_against_repository(
        args.proposal,
        args.approved_proposal_sha256,
        args.repository_root,
        dispatch_actor=args.dispatch_actor,
        triggering_actor=args.triggering_actor,
        github_run_id=args.github_run_id,
        github_run_attempt=args.github_run_attempt,
        workflow_dispatch_ref=args.workflow_dispatch_ref,
        workflow_repository_commit=args.workflow_repository_commit,
        proposal_repository_commit=args.proposal_repository_commit,
    )


def _render_proposal(args: argparse.Namespace) -> None:
    proposal = build_pending_proposal(
        args.repository_root,
        execution_repository_commit=args.execution_repository_commit,
        not_before_utc=args.not_before_utc,
        expires_at_utc=args.expires_at_utc,
    )
    payload = canonical_proposal_bytes(proposal)
    _write_exclusive(args.output, payload, maximum_bytes=ITEM12_3_MAX_PROPOSAL_BYTES)
    print(proposal.fingerprint)


def _verify_proposal(args: argparse.Namespace) -> None:
    proposal, approval = _verified(args)
    print(
        canonical_json_bytes(
            {
                "approval_fingerprint": approval.fingerprint,
                "durable_attempt_tag": durable_attempt_tag(proposal.fingerprint),
                "execution_repository_commit": proposal.execution_repository_commit,
                "proposal_sha256": proposal.fingerprint,
            }
        ).decode("utf-8")
    )


def _render_release_body(args: argparse.Namespace) -> None:
    proposal, approval = _verified(args)
    payload = durable_release_body(
        proposal,
        approval,
        proposal_repository_commit=args.proposal_repository_commit,
    )
    _write_exclusive(args.output, payload, maximum_bytes=ITEM12_3_MAX_PROPOSAL_BYTES)
    print(durable_attempt_tag(proposal.fingerprint))


def _verify_release_and_consume(args: argparse.Namespace) -> None:
    proposal, approval = _verified(args)
    release_payload = _read_json_object(args.release_json, name="release readback")
    tag_ref_payload = _read_json_object(args.tag_ref_json, name="tag readback")
    durable = verify_durable_release(
        proposal,
        approval,
        proposal_repository_commit=args.proposal_repository_commit,
        release_payload=release_payload,
        tag_ref_payload=tag_ref_payload,
        gh_release_verify_succeeded=args.gh_release_verified,
        release_creation_succeeded_in_this_run=args.release_created_in_this_run,
    )
    _write_exclusive(
        args.durable_observation_output,
        canonical_json_bytes(durable.model_dump(mode="json")),
        maximum_bytes=64 * 1024,
    )
    consumption_path, _consumption = consume_attempt(
        proposal,
        approval,
        durable,
        ledger_root=Path(proposal.execution_paths.attempt_ledger_root),
    )
    print(
        canonical_json_bytes(
            {
                "consumption_path": str(consumption_path),
                "durable_observation_path": str(
                    Path(os.path.abspath(args.durable_observation_output))  # noqa: PTH100
                ),
                "durable_consumption_fingerprint": durable.fingerprint,
            }
        ).decode("utf-8")
    )


def _issue_capability(args: argparse.Namespace) -> None:
    proposal, approval = _verified(args)
    durable = load_durable_consumption_observation(args.durable_observation)
    consumption = load_attempt_consumption(args.consumption)
    runtime_payload = load_runtime_preparation_payload(
        args.runtime_preparation_observation, proposal=proposal
    )
    capability = issue_verified_capability(
        proposal,
        approval,
        durable,
        consumption,
        runtime_payload,
    )
    _write_exclusive(
        args.capability_output,
        canonical_capability_bytes(capability),
        maximum_bytes=ITEM12_3_MAX_PROPOSAL_BYTES,
    )
    print(capability.fingerprint)


def _verify_capability(args: argparse.Namespace) -> None:
    capability = load_verified_capability(args.capability)
    verify_capability_for_execution(capability, repository_root=args.repository_root)
    print(capability.fingerprint)


def _record_terminal(args: argparse.Namespace) -> None:
    path, report = record_terminal_report(
        args.consumption,
        outcome="pre_source_failure",
        reason=args.reason,
        protected_source_acquired=False,
        stage_preserved_on_runner=False,
    )
    print(
        canonical_json_bytes({"terminal_report_path": str(path), "terminal_report": report}).decode(
            "utf-8"
        )
    )


def _render_fallback_terminal(args: argparse.Namespace) -> None:
    proposal, approval = _verified(args)
    report = build_durable_fallback_terminal(
        proposal,
        approval,
        protected_source_acquisition=args.protected_source_acquisition,
        host_source_cleanup_disposition=args.host_source_cleanup_disposition,
    )
    _write_exclusive(
        args.output,
        canonical_json_bytes(report.model_dump(mode="json")),
        maximum_bytes=4 * 1024,
    )
    print(report.fingerprint)


def _verify_terminal(args: argparse.Namespace) -> None:
    payload = args.terminal.read_bytes()
    if not payload or len(payload) > 4 * 1024:
        raise Item123AuthorizationError("terminal report violates its exact byte bound")
    report: Item123TerminalReport | Item123DurableFallbackTerminalReport
    try:
        report = Item123TerminalReport.model_validate_json(payload)
    except ValueError:
        try:
            report = Item123DurableFallbackTerminalReport.model_validate_json(payload)
        except ValueError as error:
            raise Item123AuthorizationError(
                "terminal report is outside the closed schema"
            ) from error
    if canonical_json_bytes(report.model_dump(mode="json")) != payload:
        raise Item123AuthorizationError("terminal report is not canonical JSON")
    print(report.fingerprint)


def _add_proposal_verification_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--approved-proposal-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--dispatch-actor", required=True)
    parser.add_argument("--triggering-actor", required=True)
    parser.add_argument("--github-run-id", type=int, required=True)
    parser.add_argument("--github-run-attempt", type=int, required=True)
    parser.add_argument("--workflow-dispatch-ref", required=True)
    parser.add_argument("--workflow-repository-commit", required=True)
    parser.add_argument("--proposal-repository-commit", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    render = commands.add_parser("render-proposal", help="render canonical pending bytes")
    render.add_argument("--repository-root", type=Path, required=True)
    render.add_argument("--execution-repository-commit", required=True)
    render.add_argument("--not-before-utc", required=True)
    render.add_argument("--expires-at-utc", required=True)
    render.add_argument("--output", type=Path, required=True)
    render.set_defaults(handler=_render_proposal)

    verify = commands.add_parser("verify-proposal", help="verify approved proposal and all joins")
    _add_proposal_verification_arguments(verify)
    verify.set_defaults(handler=_verify_proposal)

    body = commands.add_parser(
        "render-release-body", help="render exact immutable attempt-release body"
    )
    _add_proposal_verification_arguments(body)
    body.add_argument("--output", type=Path, required=True)
    body.set_defaults(handler=_render_release_body)

    consume = commands.add_parser(
        "verify-release-and-consume",
        help="verify durable readback, consume locally, and emit capability",
    )
    _add_proposal_verification_arguments(consume)
    consume.add_argument("--release-json", type=Path, required=True)
    consume.add_argument("--tag-ref-json", type=Path, required=True)
    consume.add_argument("--gh-release-verified", action="store_true")
    consume.add_argument("--release-created-in-this-run", action="store_true")
    consume.add_argument("--durable-observation-output", type=Path, required=True)
    consume.set_defaults(handler=_verify_release_and_consume)

    issue = commands.add_parser(
        "issue-capability",
        help="join consumed authority to the exact post-consumption runtime observation",
    )
    _add_proposal_verification_arguments(issue)
    issue.add_argument("--durable-observation", type=Path, required=True)
    issue.add_argument("--consumption", type=Path, required=True)
    issue.add_argument("--runtime-preparation-observation", type=Path, required=True)
    issue.add_argument("--capability-output", type=Path, required=True)
    issue.set_defaults(handler=_issue_capability)

    capability = commands.add_parser(
        "verify-capability", help="reconstruct C and its sealed local consumption"
    )
    capability.add_argument("--capability", type=Path, required=True)
    capability.add_argument("--repository-root", type=Path, required=True)
    capability.set_defaults(handler=_verify_capability)

    terminal = commands.add_parser(
        "record-pre-source-terminal", help="record one sanitized terminal denial"
    )
    terminal.add_argument("--consumption", type=Path, required=True)
    terminal.add_argument(
        "--reason",
        choices=("authorization_boundary_incomplete", "native_runtime_preflight_failed"),
        required=True,
    )
    terminal.set_defaults(handler=_record_terminal)

    fallback = commands.add_parser(
        "render-fallback-terminal",
        help="pre-render bounded evidence for a post-release primary-terminal failure",
    )
    _add_proposal_verification_arguments(fallback)
    fallback.add_argument(
        "--protected-source-acquisition",
        choices=(
            "unknown_after_global_consumption",
            "not_started",
            "started_incomplete",
            "completed",
        ),
        required=True,
    )
    fallback.add_argument(
        "--host-source-cleanup-disposition",
        choices=("not_applicable", "removed", "unknown"),
        required=True,
    )
    fallback.add_argument("--output", type=Path, required=True)
    fallback.set_defaults(handler=_render_fallback_terminal)

    validate_terminal = commands.add_parser(
        "verify-terminal", help="verify one exact uploadable terminal report"
    )
    validate_terminal.add_argument("--terminal", type=Path, required=True)
    validate_terminal.set_defaults(handler=_verify_terminal)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        args.handler(args)
    except Item123AuthorizationError as error:
        print(f"item12_3_authorization_denied: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception:
        # The workflow output is deliberately bounded and never includes traceback locals or paths.
        print(
            "item12_3_authorization_denied: unexpected source-free verifier failure",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
