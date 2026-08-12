"""One-use authorization contracts for the protected Item 12.3 sci-Plex3 fit.

The checked-in proposal described here is deliberately *not* execution authority.  Authority is
formed only by an exact workflow-dispatch digest, a matching canonical proposal, and an immutable
asset-free GitHub release that consumes that digest once.  A local exclusive ledger adds crash and
same-host replay protection; it is not a substitute for the durable release.

This module is source-free.  It binds the exact public source locator as approval text but never
resolves, stats, opens, or transfers the protected asset.  Only the checked-in authorized wrapper
may acquire the fixed source after capability verification and an exclusive execution-start claim.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, cast

from pydantic import ConfigDict, Field, field_validator, model_validator

from cellstate.backends.sciplex3_loader import (
    SCIPLEX3_SOURCE_BYTE_COUNT,
    SCIPLEX3_SOURCE_FILENAME,
    SCIPLEX3_SOURCE_MD5,
    SCIPLEX3_SOURCE_SHA256,
)
from cellstate.domain.common import SchemaModel, canonical_fingerprint, canonical_json_bytes
from cellstate.evaluation.sciplex3_candidate import (
    SCIPLEX3_CANDIDATE_ACTION_COUNT,
    SCIPLEX3_CANDIDATE_BATCH_SIZE,
    SCIPLEX3_CANDIDATE_FACTOR_COUNT,
    SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION,
    SCIPLEX3_CANDIDATE_MAX_OUTER_ITERATIONS,
    SCIPLEX3_CANDIDATE_MIN_OUTER_ITERATIONS,
    SCIPLEX3_CANDIDATE_MODEL_ID,
    SCIPLEX3_CANDIDATE_MODEL_SCHEMA,
    SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION,
    SCIPLEX3_CANDIDATE_OUTPUT_MODEL_SCHEMA_SHA256,
    SCIPLEX3_CANDIDATE_PLATE_COUNT,
    SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256,
    SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT,
    SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,
)
from cellstate.evaluation.sciplex3_candidate_runner import contained_training_contracts

SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"
COMMIT_PATTERN: Final = r"^[0-9a-f]{40}$"
ITEM12_3_REPOSITORY: Final = "logannye/cellstate"
ITEM12_3_REPOSITORY_OWNER: Final = "logannye"
ITEM12_3_AUTHORIZED_ACTOR: Final = "logannye"
ITEM12_3_WORKFLOW_DISPATCH_REF: Final = "refs/heads/main"
ITEM12_3_PROPOSAL_RELATIVE_PATH: Final = "audits/item12_3/sciplex3-k562-v5-pending-proposal.json"
ITEM12_3_AUTHORIZATION_MODULE_RELATIVE_PATH: Final = (
    "src/cellstate/training/item12_3_authorization.py"
)
ITEM12_3_AUTHORIZATION_CLI_RELATIVE_PATH: Final = "scripts/verify_item12_3_authorization.py"
ITEM12_3_RUNTIME_DISTRIBUTION_VERIFIER_RELATIVE_PATH: Final = (
    "scripts/verify_sciplex3_v5_runtime_distribution.py"
)
ITEM12_3_RUNTIME_OCI_VERIFIER_RELATIVE_PATH: Final = "scripts/verify_sciplex3_v5_runtime_oci.py"
ITEM12_3_WORKFLOW_RELATIVE_PATH: Final = ".github/workflows/item12-3-sciplex3-v5.yml"
ITEM12_3_AUTHORIZED_ENTRYPOINT_RELATIVE_PATH: Final = "scripts/run_item12_3_authorized_execution.py"
ITEM12_3_RUNTIME_DISTRIBUTION_LOCK_RELATIVE_PATH: Final = (
    "containers/sciplex3-v5-runtime/runtime-distribution-lock.json"
)
ITEM12_3_EXECUTION_ID: Final = "sciplex3-k562-v5-fit"
ITEM12_3_EXECUTION_ROOT: Final = "/var/lib/cellstate/item12-3"
ITEM12_3_STAGING_ROOT: Final = f"{ITEM12_3_EXECUTION_ROOT}/staging"
ITEM12_3_LEDGER_ROOT: Final = f"{ITEM12_3_EXECUTION_ROOT}/attempt-ledger"
ITEM12_3_CANONICAL_PUBLICATION_ROOT: Final = (
    "backends/vertical-a/sciplex3-k562-24h-v1/candidate-publication"
)
ITEM12_3_DOCKER_VERSION: Final = "29.7.2"
ITEM12_3_SOURCE_VERIFICATION_RELATIVE_PATH: Final = (
    "benchmarks/artifacts/sciplex3-k562-24h-v1/source-verification.json"
)
ITEM12_3_SOURCE_URI: Final = (
    "https://zenodo.org/api/records/13350497/files/SrivatsanTrapnell2020_sciplex3.h5ad/content"
)
ITEM12_3_MAX_VALIDITY_SECONDS: Final = 24 * 60 * 60
ITEM12_3_MAX_PROPOSAL_BYTES: Final = 256 * 1024
ITEM12_3_MAX_TERMINAL_REPORT_BYTES: Final = 4 * 1024

_EXPECTED_STAGE_PATHS: Final = tuple(
    sorted(
        (
            "candidate-model.json",
            "candidate-training-plan.json",
            "contained-training-observation.json",
            "contained-worker-observation.json",
            "fit/candidate-model.json",
            "fit/training-execution-observation.json",
            "materialization-manifest.json",
            "p1-assembly-receipt.json",
            "p1-finalized-count-scan-receipt.json",
            "sealed-plan/candidate-specification.json",
            "sealed-plan/candidate-training-plan.json",
            "sealed-plan/contained-execution-policy.json",
            "sealed-plan/output-model-schema.json",
            "sealed-plan/p1-count-stream-descriptor.json",
            "sealed-plan/publication-generation-seed.json",
            "sealed-plan/runtime-image-lock.json",
            "sealed-plan/runtime-lock.json",
            "sealed-plan/training-code-closure.json",
            "sealed-plan/training-execution-input-closure.json",
            "training-execution-observation.json",
        )
    )
)


def _parse_rfc3339_utc(value: str, *, name: str) -> datetime:
    if type(value) is not str:
        raise ValueError(f"{name} must be exact RFC3339 UTC text")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError(f"{name} must be exact whole-second RFC3339 UTC text") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError(f"{name} must be canonical RFC3339 UTC text")
    return parsed


class Item123AuthorizationError(RuntimeError):
    """Raised before protected-source acquisition when an authorization join fails."""


class Item123ReplayError(Item123AuthorizationError):
    """Raised when a proposal has already been consumed or may have been partially consumed."""


class Item123ToolBinding(SchemaModel):
    """One exact event-ref tool that cannot drift in the proposal-only commit."""

    model_config = ConfigDict(strict=True)

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("relative_path")
    @classmethod
    def relative_path_is_canonical(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            type(value) is not str
            or "\\" in value
            or "\x00" in value
            or path.is_absolute()
            or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("authorization tool path must be canonical repository-relative text")
        return value


class Item123CandidateBinding(SchemaModel):
    """Exact source-free scientific candidate selected for the one protected fit."""

    model_config = ConfigDict(strict=True)

    candidate_id: str
    implementation_version: Literal["5.0.0"]
    model_schema: str
    model_schema_version: Literal["5.0.0"]
    specification_sha256: str = Field(pattern=SHA256_PATTERN)
    output_model_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    factor_count: Literal[16]
    action_count: Literal[752]
    plate_count: Literal[8]
    training_record_count: Literal[94785]
    training_well_count: Literal[768]
    batch_size: Literal[512]
    optimization_seed: Literal[0]
    fixed_factor_shape_hex: Literal["0x1.999999999999ap-4"]
    minimum_outer_iterations: Literal[10]
    maximum_outer_iterations: Literal[50]


class Item123ProtectedSourceBinding(SchemaModel):
    """Exact public locator and p1 scope; no caller-supplied local path is representable."""

    model_config = ConfigDict(strict=True)

    dataset_id: Literal["sciplex3-k562-24h"]
    filename: str
    public_locator_uri: str
    accession: Literal["10.5281/zenodo.13350497"]
    sha256: str = Field(pattern=SHA256_PATTERN)
    md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    byte_count: int = Field(gt=0)
    source_verification_relative_path: Literal[
        "benchmarks/artifacts/sciplex3-k562-24h-v1/source-verification.json"
    ]
    source_verification_sha256: str = Field(pattern=SHA256_PATTERN)
    access_purpose: Literal["train_parameters"]
    partition_id: Literal["p1-train"]
    partition_role: Literal["train"]
    permitted_partitions: tuple[Literal["p1-train"], ...]
    prohibited_partitions: tuple[
        Literal["p2-calibration", "p3-model-selection-validation", "p4-untouched-test"], ...
    ]
    locator_included: Literal[True]
    local_path_included: Literal[False]
    source_path_caller_supplied: Literal[False]
    physical_asset_contains_all_partitions: Literal[True]
    entire_opaque_asset_transfer_and_snapshot_required: Literal[True]
    permitted_expression_or_raw_count_decode_scope: Literal["p1-train-rows-only"]
    full_axis_selector_metadata_decode_required: Literal[True]
    heldout_selector_metadata_resolved_and_decoded: Literal[True]
    heldout_expression_or_raw_count_values_read: Literal[False]
    heldout_expression_or_raw_count_values_decoded: Literal[False]
    heldout_endpoints_or_outcomes_resolved: Literal[False]
    heldout_rows_selected_for_training: Literal[False]
    heldout_rows_scored: Literal[False]
    heldout_rows_emitted: Literal[False]

    @model_validator(mode="after")
    def source_scope_is_exact(self) -> Item123ProtectedSourceBinding:
        if (
            self.filename != SCIPLEX3_SOURCE_FILENAME
            or self.public_locator_uri != ITEM12_3_SOURCE_URI
            or self.sha256 != SCIPLEX3_SOURCE_SHA256
            or self.md5 != SCIPLEX3_SOURCE_MD5
            or self.byte_count != SCIPLEX3_SOURCE_BYTE_COUNT
            or self.permitted_partitions != ("p1-train",)
            or self.prohibited_partitions
            != ("p2-calibration", "p3-model-selection-validation", "p4-untouched-test")
        ):
            raise ValueError("protected source binding differs from the exact p1 descriptor")
        return self


class Item123RuntimeLayerBinding(SchemaModel):
    """One ordered compressed layer in the authorized OCI child manifest."""

    model_config = ConfigDict(strict=True)

    media_type: Literal["application/vnd.oci.image.layer.v1.tar+gzip"]
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    byte_count: int = Field(gt=0)


class Item123ContainedExecutionBinding(SchemaModel):
    """Fingerprints and immutable image identity for the complete contained fit."""

    model_config = ConfigDict(strict=True)

    policy_fingerprint: str = Field(pattern=SHA256_PATTERN)
    training_code_closure_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_input_closure_sha256: str = Field(pattern=SHA256_PATTERN)
    runtime_image_lock_fingerprint: str = Field(pattern=SHA256_PATTERN)
    runtime_image_lock_file_sha256: str = Field(pattern=SHA256_PATTERN)
    runtime_image_reference: str
    runtime_image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    runtime_archive_sha256: str = Field(pattern=SHA256_PATTERN)
    runtime_oci_index_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    runtime_config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    runtime_layers: tuple[Item123RuntimeLayerBinding, ...] = Field(min_length=1)
    worker_command: tuple[str, ...] = Field(min_length=1)


class Item123RuntimeDistributionBinding(SchemaModel):
    """Exact immutable distribution and native-daemon identity verified before source."""

    model_config = ConfigDict(strict=True)

    distribution_lock_sha256: str = Field(pattern=SHA256_PATTERN)
    repository: Literal["logannye/cellstate"]
    release_tag: str
    release_target_commit: str = Field(pattern=COMMIT_PATTERN)
    release_target_role: Literal["immutable-runtime-dependency-provenance-not-execution-code"]
    release_target_equals_execution_commit_required: Literal[False]
    asset_name: str
    asset_sha256: str = Field(pattern=SHA256_PATTERN)
    asset_byte_count: int = Field(gt=0)
    qualified_asset_url: str
    attestation_predicate_type: Literal["https://in-toto.io/attestation/release/v0.2"]
    oci_index_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    runtime_image_lock_sha256: str = Field(pattern=SHA256_PATTERN)
    provider: Literal["github-release-asset"]
    immutable_release_required: Literal[True]
    release_attestation_required: Literal[True]
    asset_attestation_required: Literal[True]


class Item123NativeRuntimeBinding(SchemaModel):
    """Exact native host boundary; emulation and remote daemons are not authorized."""

    model_config = ConfigDict(strict=True)

    runner_operating_system: Literal["linux"]
    runner_architecture: Literal["x86_64"]
    host_operating_system: Literal["linux"]
    host_architecture: Literal["x86_64"]
    docker_server_version: Literal["29.7.2"]
    docker_operating_system: Literal["linux"]
    docker_architecture: Literal["x86_64"]
    image_store_status: tuple[Literal["driver-type"], Literal["io.containerd.snapshotter.v1"]]
    cgroup_version: Literal["2"]
    memory_limit_supported: Literal[True]
    memory_swap_limit_supported: Literal[True]
    pids_limit_supported: Literal[True]
    local_unix_socket_required: Literal[True]
    emulation_permitted: Literal[False]


class Item123ResourceLimits(SchemaModel):
    """Exact aggregate resource gates inherited from the contained policy."""

    model_config = ConfigDict(strict=True)

    parent_wall_clock_seconds: int = Field(gt=0)
    protected_source_watchdog_seconds: int = Field(gt=0)
    cleanup_timeout_seconds: int = Field(gt=0)
    memory_max_bytes: int = Field(gt=0)
    memory_swap_max_bytes: int = Field(gt=0)
    pids_limit: int = Field(gt=1)
    temporary_max_bytes: int = Field(gt=0)
    snapshot_max_bytes: int = Field(gt=0)
    workflow_timeout_minutes: Literal[150]
    runtime_asset_connect_timeout_seconds: Literal[10]
    runtime_asset_total_timeout_seconds: Literal[300]
    runtime_asset_retry_count: Literal[0]
    source_download_connect_timeout_seconds: Literal[30]
    source_download_total_timeout_seconds: Literal[1800]
    source_download_retry_count: Literal[0]
    protected_source_cleanup_bounded_by_workflow_timeout: Literal[True]
    terminal_fallback_timeout_seconds: Literal[60]

    @model_validator(mode="after")
    def limits_are_coherent(self) -> Item123ResourceLimits:
        if (
            self.memory_swap_max_bytes != self.memory_max_bytes
            or self.protected_source_watchdog_seconds >= self.parent_wall_clock_seconds
            or self.temporary_max_bytes > self.memory_max_bytes
        ):
            raise ValueError("Item 12.3 resource gates are incoherent")
        return self


class Item123ExecutionPaths(SchemaModel):
    """Fixed host roots and the exact stage-only output allowlist."""

    model_config = ConfigDict(strict=True)

    execution_id: Literal["sciplex3-k562-v5-fit"]
    execution_root: Literal["/var/lib/cellstate/item12-3"]
    staging_root: Literal["/var/lib/cellstate/item12-3/staging"]
    attempt_ledger_root: Literal["/var/lib/cellstate/item12-3/attempt-ledger"]
    runtime_archive_path: Literal[
        "/var/lib/cellstate/item12-3/runtime/"
        "sciplex3-v5-runtime-linux-amd64-"
        "37c2fa5846acfbd8357476859bd7f8f0ac6591261d79c2f6f46f0aa22fb76454.oci.tar"
    ]
    protected_source_path: Literal[
        "/var/lib/cellstate/item12-3/protected-source/SrivatsanTrapnell2020_sciplex3.h5ad"
    ]
    canonical_publication_root: Literal[
        "backends/vertical-a/sciplex3-k562-24h-v1/candidate-publication"
    ]
    permitted_stage_relative_paths: tuple[str, ...] = Field(min_length=1)
    parent_terminal_report_relative_path: Literal["contained-training-terminal-observation.json"]
    parent_terminal_report_max_bytes: Literal[65536]
    sanitized_terminal_report_filename: Literal["item12-3-terminal-report.json"]
    sanitized_terminal_report_max_bytes: Literal[4096]
    sanitized_terminal_artifact_retention_days: Literal[90]
    durable_terminal_scope: Literal["generic-unknown-state-fallback-in-immutable-attempt-release"]
    actual_terminal_outcome_durability: Literal[
        "actions-artifact-90-days-then-exact-reviewed-repository-commit"
    ]
    authorized_post_run_terminal_repository_path: Literal[
        "audits/item12_3/sciplex3-k562-v5-terminal.json"
    ]
    source_acquisition_started_marker: Literal[
        "/var/lib/cellstate/item12-3/source-acquisition-started.json"
    ]
    source_acquisition_completed_marker: Literal[
        "/var/lib/cellstate/item12-3/source-acquisition-completed.json"
    ]
    host_source_removed_marker: Literal["/var/lib/cellstate/item12-3/host-source-removed.json"]

    @model_validator(mode="after")
    def output_allowlist_is_exact(self) -> Item123ExecutionPaths:
        if self.permitted_stage_relative_paths != _EXPECTED_STAGE_PATHS:
            raise ValueError("Item 12.3 stage output allowlist is not exact")
        return self


class Item123Prohibitions(SchemaModel):
    """Negative authority that survives both successful and failed fitting."""

    model_config = ConfigDict(strict=True)

    source_persistence_permitted: Literal[False]
    source_upload_permitted: Literal[False]
    staged_output_upload_permitted: Literal[False]
    model_upload_permitted: Literal[False]
    canonical_publication_permitted: Literal[False]
    runtime_builder_permitted: Literal[False]
    p2_p3_p4_access_permitted: Literal[False]
    evaluation_or_calibration_permitted: Literal[False]
    lifecycle_evidence_permitted: Literal[False]
    scientific_admission_permitted: Literal[False]
    retry_permitted: Literal[False]
    resume_permitted: Literal[False]
    only_sanitized_terminal_report_may_leave_runner: Literal[True]
    stage_and_model_destroyed_with_ephemeral_runner_after_terminalization: Literal[True]
    exact_sanitized_terminal_repository_persistence_permitted: Literal[True]
    terminal_persistence_grants_lifecycle_or_scientific_authority: Literal[False]


class Item123StopPolicy(SchemaModel):
    """A dispatch is terminal after one consumption, regardless of outcome."""

    model_config = ConfigDict(strict=True)

    maximum_attempt_count: Literal[1]
    consume_before_protected_source_acquisition: Literal[True]
    stop_after_authorization_failure: Literal[True]
    stop_after_preflight_failure: Literal[True]
    stop_after_runtime_failure: Literal[True]
    stop_after_successful_stage: Literal[True]
    no_automatic_retry: Literal[True]
    no_manual_reuse: Literal[True]
    hard_runner_death_may_leave_only_immutable_generic_fallback: Literal[True]


class Item123ExecutionProposal(SchemaModel):
    """Canonical pending proposal; these bytes explicitly grant no authority on their own."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-item12-3-execution-proposal"] = (
        "cellstate-item12-3-execution-proposal"
    )
    artifact_schema_version: Literal["1.1.0"] = "1.1.0"
    status: Literal["pending-explicit-user-approval"]
    grants_execution_authority: Literal[False]
    approval_mechanism: Literal[
        "workflow-dispatch-exact-proposal-sha256-plus-immutable-one-use-release"
    ]
    repository: Literal["logannye/cellstate"]
    repository_owner: Literal["logannye"]
    authorized_dispatch_actor: Literal["logannye"]
    workflow_dispatch_ref: Literal["refs/heads/main"]
    execution_repository_commit: str = Field(pattern=COMMIT_PATTERN)
    not_before_utc: str
    expires_at_utc: str
    proposal_only_commit_required: Literal[True]
    candidate: Item123CandidateBinding
    protected_source: Item123ProtectedSourceBinding
    contained_execution: Item123ContainedExecutionBinding
    runtime_distribution: Item123RuntimeDistributionBinding
    native_runtime: Item123NativeRuntimeBinding
    resource_limits: Item123ResourceLimits
    execution_paths: Item123ExecutionPaths
    prohibitions: Item123Prohibitions
    stop_policy: Item123StopPolicy
    authorization_tools: tuple[Item123ToolBinding, ...] = Field(min_length=6)
    authorized_entrypoint: Literal["scripts/run_item12_3_authorized_execution.py"]

    @model_validator(mode="after")
    def joins_are_exact(self) -> Item123ExecutionProposal:
        not_before = _parse_rfc3339_utc(self.not_before_utc, name="proposal not-before")
        expires_at = _parse_rfc3339_utc(self.expires_at_utc, name="proposal expiry")
        validity_seconds = (expires_at - not_before).total_seconds()
        if not 0 < validity_seconds <= ITEM12_3_MAX_VALIDITY_SECONDS:
            raise ValueError("proposal validity interval must be positive and at most 24 hours")
        tool_paths = tuple(binding.relative_path for binding in self.authorization_tools)
        expected_tools = tuple(
            sorted(
                (
                    ITEM12_3_AUTHORIZATION_CLI_RELATIVE_PATH,
                    ITEM12_3_AUTHORIZATION_MODULE_RELATIVE_PATH,
                    ITEM12_3_AUTHORIZED_ENTRYPOINT_RELATIVE_PATH,
                    ITEM12_3_RUNTIME_DISTRIBUTION_VERIFIER_RELATIVE_PATH,
                    ITEM12_3_RUNTIME_OCI_VERIFIER_RELATIVE_PATH,
                    ITEM12_3_WORKFLOW_RELATIVE_PATH,
                )
            )
        )
        if tool_paths != expected_tools or len(tool_paths) != len(set(tool_paths)):
            raise ValueError("authorization tooling closure is incomplete or unordered")
        contained = self.contained_execution
        distribution = self.runtime_distribution
        if (
            contained.runtime_archive_sha256 != distribution.asset_sha256
            or contained.runtime_oci_index_digest != distribution.oci_index_digest
            or contained.runtime_image_digest != distribution.image_digest
            or contained.runtime_config_digest != distribution.config_digest
            or contained.runtime_image_lock_file_sha256 != distribution.runtime_image_lock_sha256
            or distribution.repository != self.repository
            or self.native_runtime.docker_server_version != ITEM12_3_DOCKER_VERSION
            or PurePosixPath(self.execution_paths.runtime_archive_path).name
            != distribution.asset_name
            or PurePosixPath(self.execution_paths.protected_source_path).name
            != self.protected_source.filename
        ):
            raise ValueError("proposal runtime, image, and distribution joins disagree")
        candidate = self.candidate
        if (
            candidate.candidate_id != SCIPLEX3_CANDIDATE_MODEL_ID
            or candidate.implementation_version != SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION
            or candidate.model_schema != SCIPLEX3_CANDIDATE_MODEL_SCHEMA
            or candidate.model_schema_version != SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION
            or candidate.specification_sha256 != SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256
            or candidate.output_model_schema_sha256 != SCIPLEX3_CANDIDATE_OUTPUT_MODEL_SCHEMA_SHA256
        ):
            raise ValueError("proposal candidate differs from the frozen v5 candidate")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


class Item123DispatchApproval(SchemaModel):
    """Stable interpretation of the exact digest typed at workflow dispatch."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-item12-3-dispatch-approval"] = (
        "cellstate-item12-3-dispatch-approval"
    )
    artifact_schema_version: Literal["1.1.0"] = "1.1.0"
    approved_proposal_sha256: str = Field(pattern=SHA256_PATTERN)
    dispatch_actor: Literal["logannye"]
    triggering_actor: Literal["logannye"]
    github_run_id: int = Field(gt=0)
    github_run_attempt: Literal[1]
    workflow_dispatch_ref: Literal["refs/heads/main"]
    workflow_repository_commit: str = Field(pattern=COMMIT_PATTERN)
    proposal_repository_commit: str = Field(pattern=COMMIT_PATTERN)
    mechanism: Literal["workflow-dispatch-exact-proposal-sha256-plus-immutable-one-use-release"] = (
        "workflow-dispatch-exact-proposal-sha256-plus-immutable-one-use-release"
    )
    acknowledges_pending_proposal_is_not_authority: Literal[True] = True
    approves_one_attempt_only: Literal[True] = True
    retry_permitted: Literal[False] = False

    @model_validator(mode="after")
    def workflow_and_proposal_commits_are_distinct(self) -> Item123DispatchApproval:
        if self.workflow_repository_commit == self.proposal_repository_commit:
            raise ValueError("proposal D must be distinct from workflow/execution C")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


class Item123DurableReleaseObservation(SchemaModel):
    """Readback of the immutable, asset-free release that globally consumes a proposal."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-item12-3-durable-consumption"] = (
        "cellstate-item12-3-durable-consumption"
    )
    artifact_schema_version: Literal["1.1.0"] = "1.1.0"
    repository: Literal["logannye/cellstate"]
    dispatch_actor: Literal["logannye"]
    triggering_actor: Literal["logannye"]
    github_run_id: int = Field(gt=0)
    github_run_attempt: Literal[1]
    workflow_dispatch_ref: Literal["refs/heads/main"]
    tag_name: str
    workflow_repository_commit: str = Field(pattern=COMMIT_PATTERN)
    proposal_repository_commit: str = Field(pattern=COMMIT_PATTERN)
    proposal_sha256: str = Field(pattern=SHA256_PATTERN)
    proposal_fingerprint: str = Field(pattern=SHA256_PATTERN)
    approval_fingerprint: str = Field(pattern=SHA256_PATTERN)
    execution_repository_commit: str = Field(pattern=COMMIT_PATTERN)
    execution_id: Literal["sciplex3-k562-v5-fit"]
    fallback_terminal_sha256: str = Field(pattern=SHA256_PATTERN)
    release_id: int = Field(gt=0)
    release_is_draft: Literal[False]
    release_is_prerelease: Literal[False]
    release_is_immutable: Literal[True]
    release_asset_count: Literal[0]
    tag_target_type: Literal["commit"]
    gh_release_verify_succeeded: Literal[True]
    release_creation_succeeded_in_this_run: Literal[True]
    no_retry: Literal[True]

    @model_validator(mode="after")
    def commit_roles_are_exact(self) -> Item123DurableReleaseObservation:
        if self.workflow_repository_commit != self.execution_repository_commit:
            raise ValueError("durable workflow commit must equal execution commit C")
        if self.proposal_repository_commit == self.workflow_repository_commit:
            raise ValueError("durable proposal D must be distinct from workflow/execution C")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


class Item123AttemptConsumption(SchemaModel):
    """Runner-local immutable record written before source acquisition."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-item12-3-attempt-consumption"] = (
        "cellstate-item12-3-attempt-consumption"
    )
    artifact_schema_version: Literal["1.0.0"] = "1.0.0"
    attempt_key: str = Field(pattern=SHA256_PATTERN)
    proposal_sha256: str = Field(pattern=SHA256_PATTERN)
    proposal_fingerprint: str = Field(pattern=SHA256_PATTERN)
    approval_fingerprint: str = Field(pattern=SHA256_PATTERN)
    durable_consumption_fingerprint: str = Field(pattern=SHA256_PATTERN)
    execution_repository_commit: str = Field(pattern=COMMIT_PATTERN)
    execution_id: Literal["sciplex3-k562-v5-fit"]
    protected_source_acquired_at_consumption: Literal[False]
    attempt_count: Literal[1]
    no_retry: Literal[True]

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


class Item123ExecutionStart(SchemaModel):
    """Exclusive runner-local claim that prevents concurrent capability invocation."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-item12-3-execution-start"] = (
        "cellstate-item12-3-execution-start"
    )
    artifact_schema_version: Literal["1.0.0"] = "1.0.0"
    attempt_key: str = Field(pattern=SHA256_PATTERN)
    attempt_consumption_fingerprint: str = Field(pattern=SHA256_PATTERN)
    capability_fingerprint: str = Field(pattern=SHA256_PATTERN)
    proposal_sha256: str = Field(pattern=SHA256_PATTERN)
    github_run_id: int = Field(gt=0)
    github_run_attempt: Literal[1]
    execution_id: Literal["sciplex3-k562-v5-fit"]
    protected_source_acquired_at_start: Literal[False]
    retry_permitted: Literal[False]

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


class VerifiedItem123ExecutionCapability(SchemaModel):
    """Canonical pre-source receipt proving that the exact one-use attempt was consumed.

    This is a transport object for the checked-in post-approval wrapper, not a bearer token.  The
    wrapper must call :func:`verify_capability_for_execution`, which reconstructs the proposal and
    re-reads the fixed local consumption marker before it is allowed to resolve protected source.
    """

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-item12-3-verified-execution-capability"] = (
        "cellstate-item12-3-verified-execution-capability"
    )
    artifact_schema_version: Literal["1.1.0"] = "1.1.0"
    capability_state: Literal["verified-consumed-one-use"]
    proposal: Item123ExecutionProposal
    proposal_canonical_json: str = Field(min_length=2, max_length=ITEM12_3_MAX_PROPOSAL_BYTES)
    proposal_sha256: str = Field(pattern=SHA256_PATTERN)
    approval_fingerprint: str = Field(pattern=SHA256_PATTERN)
    durable_consumption: Item123DurableReleaseObservation
    attempt_consumption: Item123AttemptConsumption
    runtime_preparation_canonical_json: str = Field(min_length=2, max_length=64 * 1024)
    runtime_preparation_observation_sha256: str = Field(pattern=SHA256_PATTERN)
    protected_source_acquired_at_issue: Literal[False]
    retry_permitted: Literal[False]

    @model_validator(mode="after")
    def receipt_joins_are_exact(self) -> VerifiedItem123ExecutionCapability:
        proposal_bytes = canonical_proposal_bytes(self.proposal)
        attempt = self.attempt_consumption
        durable = self.durable_consumption
        runtime_payload = self.runtime_preparation_canonical_json.encode("utf-8")
        if (
            self.proposal_canonical_json.encode("utf-8") != proposal_bytes
            or self.proposal_sha256 != _sha256(proposal_bytes)
            or self.proposal_sha256 != self.proposal.fingerprint
            or self.approval_fingerprint != durable.approval_fingerprint
            or self.approval_fingerprint != attempt.approval_fingerprint
            or durable.proposal_sha256 != self.proposal_sha256
            or durable.proposal_fingerprint != self.proposal.fingerprint
            or attempt.proposal_sha256 != self.proposal_sha256
            or attempt.proposal_fingerprint != self.proposal.fingerprint
            or attempt.durable_consumption_fingerprint != durable.fingerprint
            or durable.execution_repository_commit != self.proposal.execution_repository_commit
            or attempt.execution_repository_commit != self.proposal.execution_repository_commit
            or durable.execution_id != self.proposal.execution_paths.execution_id
            or attempt.execution_id != self.proposal.execution_paths.execution_id
            or _sha256(runtime_payload) != self.runtime_preparation_observation_sha256
        ):
            raise ValueError("verified execution capability has a broken typed join")
        _validate_runtime_preparation_observation(self.proposal, runtime_payload)
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


TerminalOutcome = Literal["success", "pre_source_failure", "runtime_failure"]
TerminalReason = Literal[
    "stage_sealed",
    "authorization_boundary_incomplete",
    "native_runtime_preflight_failed",
    "runtime_distribution_failed",
    "protected_source_acquisition_failed",
    "protected_source_cleanup_failed",
    "contained_execution_failed",
    "terminal_evidence_failed",
]
ContainedTerminalStatus = Literal[
    "not_started",
    "success",
    "timeout",
    "oom_killed",
    "worker_failure",
    "stage_rejected",
    "supervisor_failure",
]
ContainedFailureCode = Literal[
    "not_applicable",
    "none",
    "worker_timed_out",
    "worker_oom_killed",
    "worker_exited_nonzero",
    "worker_report_missing",
    "worker_report_invalid",
    "worker_report_contradiction",
    "stage_inventory_invalid",
    "stage_semantic_verification_failed",
    "canonical_publication_changed",
    "canonical_publication_identity_invalid",
    "contained_executor_failed",
]


class Item123TerminalReport(SchemaModel):
    """Bounded terminal report; it cannot carry paths, logs, source, model, or stage bytes."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-item12-3-terminal-report"] = (
        "cellstate-item12-3-terminal-report"
    )
    artifact_schema_version: Literal["1.1.0"] = "1.1.0"
    attempt_key: str = Field(pattern=SHA256_PATTERN)
    attempt_consumption_fingerprint: str = Field(pattern=SHA256_PATTERN)
    proposal_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_id: Literal["sciplex3-k562-v5-fit"]
    outcome: TerminalOutcome
    reason: TerminalReason
    contained_terminal_status: ContainedTerminalStatus
    contained_failure_code: ContainedFailureCode
    execution_outcome: Literal[
        "not_started", "unavailable", "success", "timeout", "oom_killed", "worker_failure"
    ]
    parent_wall_clock_elapsed_seconds: float | None = Field(default=None, ge=0.0)
    aggregate_container_limits_enforced: bool
    runtime_preparation_observation_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    protected_source_acquired: bool
    protected_source_acquisition: Literal["not_started", "started_incomplete", "completed"]
    protected_source_cleanup_disposition: Literal["not_applicable", "removed", "failed", "unknown"]
    host_source_cleanup_disposition: Literal["not_applicable", "removed", "failed"]
    container_cleanup_disposition: Literal["not_started", "proved_removed", "unproved"]
    snapshot_volume_cleanup_disposition: Literal["not_started", "proved_removed", "unproved"]
    source_match_disposition: Literal[
        "not_acquired",
        "acquired_unverified",
        "pre_and_post_match",
        "pre_match_post_failed",
        "mismatch",
    ]
    worker_report_status: Literal["unknown", "not_started", "verified", "missing", "invalid"]
    stage_disposition: Literal["unknown", "not_created", "sealed", "quarantined"]
    stage_semantic_verification: Literal["unknown", "not_attempted", "verified", "failed"]
    canonical_publication_unchanged: bool | None = None
    parent_terminal_observation_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    staged_success_observation_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    staged_tree_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    stage_preserved_on_runner: bool | None
    canonical_publication_performed: Literal[False]
    nonterminal_artifact_uploaded: Literal[False]
    terminal_report_upload_permitted: Literal[True]
    uploaded_payload_scope: Literal["terminal-report-only"]
    retry_permitted: Literal[False]

    @model_validator(mode="after")
    def success_requires_exact_stage_evidence(self) -> Item123TerminalReport:
        if self.outcome == "success":
            if (
                self.reason != "stage_sealed"
                or not self.protected_source_acquired
                or not self.stage_preserved_on_runner
                or self.parent_terminal_observation_sha256 is None
                or self.staged_success_observation_sha256 is None
                or self.staged_tree_sha256 is None
            ):
                raise ValueError("successful terminal report lacks sealed-stage evidence")
        elif self.reason == "stage_sealed":
            raise ValueError("failed terminal report cannot claim a sealed stage")
        if self.outcome == "pre_source_failure" and self.protected_source_acquired:
            raise ValueError("pre-source terminal failure cannot claim source acquisition")
        if self.protected_source_acquired != (self.protected_source_acquisition != "not_started"):
            raise ValueError("terminal source acquisition state contradicts its started flag")
        if (self.protected_source_acquisition == "not_started") != (
            self.protected_source_cleanup_disposition == "not_applicable"
        ):
            raise ValueError("terminal source cleanup disposition contradicts acquisition")
        if self.protected_source_acquisition == "not_started":
            if (
                self.host_source_cleanup_disposition != "not_applicable"
                or self.container_cleanup_disposition != "not_started"
                or self.snapshot_volume_cleanup_disposition != "not_started"
            ):
                raise ValueError("pre-source terminal report invents cleanup evidence")
        elif self.protected_source_cleanup_disposition == "removed":
            if self.host_source_cleanup_disposition != "removed" or (
                self.container_cleanup_disposition,
                self.snapshot_volume_cleanup_disposition,
            ) not in {
                ("not_started", "not_started"),
                ("proved_removed", "proved_removed"),
            }:
                raise ValueError("removed source cleanup lacks complete cleanup proof")
        elif (
            self.host_source_cleanup_disposition == "failed"
            and self.protected_source_cleanup_disposition != "failed"
        ):
            raise ValueError("failed host cleanup cannot be reported as unknown")
        if self.protected_source_acquired != (self.source_match_disposition != "not_acquired"):
            raise ValueError("terminal source acquisition and identity disposition disagree")
        if self.protected_source_acquired and self.runtime_preparation_observation_sha256 is None:
            raise ValueError("source acquisition began without verified runtime preparation")
        if self.outcome == "pre_source_failure" and (
            self.contained_terminal_status != "not_started"
            or self.contained_failure_code != "not_applicable"
            or self.execution_outcome != "not_started"
            or self.parent_wall_clock_elapsed_seconds is not None
            or self.aggregate_container_limits_enforced
            or self.runtime_preparation_observation_sha256 is not None
            or self.protected_source_acquisition != "not_started"
            or self.protected_source_cleanup_disposition != "not_applicable"
            or self.host_source_cleanup_disposition != "not_applicable"
            or self.container_cleanup_disposition != "not_started"
            or self.snapshot_volume_cleanup_disposition != "not_started"
            or self.source_match_disposition != "not_acquired"
            or self.worker_report_status != "not_started"
            or self.stage_disposition != "not_created"
            or self.stage_semantic_verification != "not_attempted"
            or self.canonical_publication_unchanged is not None
            or self.parent_terminal_observation_sha256 is not None
            or self.staged_success_observation_sha256 is not None
            or self.staged_tree_sha256 is not None
            or self.stage_preserved_on_runner is not False
        ):
            raise ValueError("pre-source terminal report claims contained execution evidence")
        if self.outcome == "success" and (
            self.contained_terminal_status != "success"
            or self.contained_failure_code != "none"
            or self.execution_outcome != "success"
            or self.parent_wall_clock_elapsed_seconds is None
            or not self.aggregate_container_limits_enforced
            or self.protected_source_acquisition != "completed"
            or self.protected_source_cleanup_disposition != "removed"
            or self.source_match_disposition != "pre_and_post_match"
            or self.worker_report_status != "verified"
            or self.stage_disposition != "sealed"
            or self.stage_semantic_verification != "verified"
            or self.canonical_publication_unchanged is not True
        ):
            raise ValueError("successful terminal report contradicts contained evidence")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


class Item123DurableFallbackTerminalReport(SchemaModel):
    """Pre-rendered bounded evidence used only if post-release primary terminalization fails."""

    model_config = ConfigDict(strict=True)

    artifact_schema: Literal["cellstate-item12-3-durable-fallback-terminal-report"] = (
        "cellstate-item12-3-durable-fallback-terminal-report"
    )
    artifact_schema_version: Literal["1.1.0"] = "1.1.0"
    proposal_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_repository_commit: str = Field(pattern=COMMIT_PATTERN)
    workflow_repository_commit: str = Field(pattern=COMMIT_PATTERN)
    proposal_repository_commit: str = Field(pattern=COMMIT_PATTERN)
    workflow_dispatch_ref: Literal["refs/heads/main"]
    attempt_tag: str
    dispatch_actor: Literal["logannye"]
    triggering_actor: Literal["logannye"]
    github_run_id: int = Field(gt=0)
    github_run_attempt: Literal[1]
    outcome: Literal["runtime_failure"]
    reason: Literal["terminal_evidence_failed"]
    durable_consumption_disposition: Literal["creation-attempted-no-retry"]
    runtime_preparation_disposition: Literal["unavailable"]
    protected_source_acquired: bool | None
    protected_source_acquisition: Literal[
        "unknown_after_global_consumption", "not_started", "started_incomplete", "completed"
    ]
    protected_source_cleanup_disposition: Literal["not_applicable", "unknown"]
    host_source_cleanup_disposition: Literal["not_applicable", "removed", "unknown"]
    container_cleanup_disposition: Literal["not_started", "unknown"]
    snapshot_volume_cleanup_disposition: Literal["not_started", "unknown"]
    source_match_disposition: Literal["unknown", "not_acquired", "acquired_unverified"]
    canonical_publication_performed: Literal[False]
    nonterminal_artifact_uploaded: Literal[False]
    terminal_report_upload_permitted: Literal[True]
    uploaded_payload_scope: Literal["terminal-report-only"]
    retry_permitted: Literal[False]

    @model_validator(mode="after")
    def source_state_is_conservative(self) -> Item123DurableFallbackTerminalReport:
        if self.workflow_repository_commit != self.execution_repository_commit:
            raise ValueError("fallback workflow commit must equal execution commit C")
        if self.proposal_repository_commit == self.workflow_repository_commit:
            raise ValueError("fallback proposal D must be distinct from workflow/execution C")
        if self.protected_source_acquisition == "unknown_after_global_consumption":
            if (
                self.protected_source_acquired is not None
                or self.protected_source_cleanup_disposition != "unknown"
                or self.host_source_cleanup_disposition != "unknown"
                or self.container_cleanup_disposition != "unknown"
                or self.snapshot_volume_cleanup_disposition != "unknown"
                or self.source_match_disposition != "unknown"
            ):
                raise ValueError("unknown fallback source state makes a specific claim")
            return self
        acquired = self.protected_source_acquisition != "not_started"
        if self.protected_source_acquired != acquired:
            raise ValueError("fallback source acquisition state is contradictory")
        if acquired != (self.source_match_disposition == "acquired_unverified"):
            raise ValueError("fallback source identity state is contradictory")
        if (not acquired) != (self.protected_source_cleanup_disposition == "not_applicable"):
            raise ValueError("fallback source cleanup state is contradictory")
        if acquired:
            if (
                self.protected_source_cleanup_disposition != "unknown"
                or self.container_cleanup_disposition != "unknown"
                or self.snapshot_volume_cleanup_disposition != "unknown"
            ):
                raise ValueError("fallback cannot prove complete contained cleanup")
        elif (
            self.host_source_cleanup_disposition != "not_applicable"
            or self.container_cleanup_disposition != "not_started"
            or self.snapshot_volume_cleanup_disposition != "not_started"
        ):
            raise ValueError("pre-source fallback invents cleanup activity")
        return self

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _exact_sha256(value: str, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Item123AuthorizationError(f"{name} must be an exact lowercase SHA-256")
    return value


def _exact_commit(value: str, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Item123AuthorizationError(f"{name} must be an exact lowercase commit ID")
    return value


def _read_stable_regular_file(path: Path, *, name: str, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise Item123AuthorizationError(f"{name} must be one non-hardlinked regular file")
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Item123AuthorizationError(f"cannot open {name}") from error
    try:
        opened = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened.st_size < 0
            or opened.st_size > maximum_bytes
        ):
            raise Item123AuthorizationError(f"{name} changed or exceeds its size bound")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise Item123AuthorizationError(f"{name} ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise Item123AuthorizationError(f"{name} grew while being read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (opened.st_dev, opened.st_ino, opened.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise Item123AuthorizationError(f"{name} changed while being read")
    return b"".join(chunks)


def _repository_file(repository_root: Path, relative_path: str, *, name: str) -> bytes:
    root = Path(os.path.abspath(repository_root))  # noqa: PTH100 - lexical, no resolution
    try:
        root_state = root.lstat()
    except OSError as error:
        raise Item123AuthorizationError("cannot inspect repository root") from error
    if not stat.S_ISDIR(root_state.st_mode) or stat.S_ISLNK(root_state.st_mode):
        raise Item123AuthorizationError("repository root must be one real directory")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise Item123AuthorizationError(f"{name} path is not repository-relative")
    current = root
    for part in pure.parts[:-1]:
        current = current / part
        try:
            state = current.lstat()
        except OSError as error:
            raise Item123AuthorizationError(f"cannot inspect {name} parent") from error
        if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
            raise Item123AuthorizationError(f"{name} parent must be one real directory")
    return _read_stable_regular_file(
        root.joinpath(*pure.parts), name=name, maximum_bytes=16 * 1024 * 1024
    )


def _git_output(repository_root: Path, arguments: tuple[str, ...]) -> bytes:
    environment = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    try:
        result = subprocess.run(
            ("git", "-c", "core.hooksPath=/dev/null", "-C", str(repository_root), *arguments),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10.0,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise Item123AuthorizationError("cannot authenticate execution repository state") from error
    if result.returncode != 0 or len(result.stdout) > 1024 * 1024:
        raise Item123AuthorizationError("cannot authenticate execution repository state")
    return result.stdout


def _verify_clean_execution_checkout(repository_root: Path, *, expected_commit: str) -> None:
    root = Path(os.path.abspath(repository_root))  # noqa: PTH100 - lexical, no resolution
    try:
        state = root.lstat()
    except OSError as error:
        raise Item123AuthorizationError("execution repository root is unavailable") from error
    if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise Item123AuthorizationError("execution repository root must be one real directory")
    observed_head = (
        _git_output(root, ("rev-parse", "--verify", "HEAD"))
        .decode("ascii", errors="strict")
        .strip()
    )
    if observed_head != _exact_commit(expected_commit, name="execution repository commit"):
        raise Item123AuthorizationError("execution repository HEAD differs from approved commit C")
    status = _git_output(
        root,
        ("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"),
    )
    if status:
        raise Item123AuthorizationError("execution repository has tracked or untracked drift")


def _verify_proposal_only_commit(
    repository_root: Path,
    proposal: Item123ExecutionProposal,
    *,
    proposal_repository_commit: str,
) -> None:
    """Prove data-only D adds the exact approved proposal atop the checked-out trusted C."""

    root = Path(os.path.abspath(repository_root))  # noqa: PTH100 - lexical, no resolution
    proposal_commit = _exact_commit(proposal_repository_commit, name="proposal repository commit")
    head = _git_output(root, ("rev-parse", "--verify", "HEAD")).decode("ascii").strip()
    parents = (
        _git_output(root, ("rev-list", "--parents", "-n", "1", proposal_commit))
        .decode("ascii")
        .split()
    )
    if head != proposal.execution_repository_commit:
        raise Item123AuthorizationError("trusted workflow checkout is not execution commit C")
    if parents != [proposal_commit, proposal.execution_repository_commit]:
        raise Item123AuthorizationError("proposal D is not the one-parent child of execution C")
    expected_diff = f"A\t{ITEM12_3_PROPOSAL_RELATIVE_PATH}\n".encode()
    observed_diff = _git_output(
        root,
        (
            "diff-tree",
            "--no-renames",
            "--no-commit-id",
            "--name-status",
            "-r",
            proposal.execution_repository_commit,
            proposal_commit,
        ),
    )
    if observed_diff != expected_diff:
        raise Item123AuthorizationError("proposal D changes more than the one proposal blob")
    if _git_output(
        root,
        (
            "ls-tree",
            "--name-only",
            proposal.execution_repository_commit,
            "--",
            ITEM12_3_PROPOSAL_RELATIVE_PATH,
        ),
    ):
        raise Item123AuthorizationError("proposal already exists in execution commit C")
    tree_entry = _git_output(
        root,
        ("ls-tree", proposal_commit, "--", ITEM12_3_PROPOSAL_RELATIVE_PATH),
    )
    entry_prefix = b"100644 blob "
    entry_suffix = f"\t{ITEM12_3_PROPOSAL_RELATIVE_PATH}\n".encode()
    if (
        not tree_entry.startswith(entry_prefix)
        or not tree_entry.endswith(entry_suffix)
        or len(tree_entry) != len(entry_prefix) + 40 + len(entry_suffix)
    ):
        raise Item123AuthorizationError("proposal D path is not one exact 100644 blob")
    object_spec = f"{proposal_commit}:{ITEM12_3_PROPOSAL_RELATIVE_PATH}"
    if _git_output(root, ("cat-file", "-t", object_spec)) != b"blob\n":
        raise Item123AuthorizationError("proposal D path is not one blob")
    try:
        blob_size = int(_git_output(root, ("cat-file", "-s", object_spec)).strip())
    except ValueError as error:
        raise Item123AuthorizationError("proposal D blob size is malformed") from error
    if not 2 <= blob_size <= ITEM12_3_MAX_PROPOSAL_BYTES:
        raise Item123AuthorizationError("proposal D blob violates its size bound")
    repository_proposal = _git_output(root, ("cat-file", "blob", object_spec))
    if len(repository_proposal) != blob_size:
        raise Item123AuthorizationError("proposal D blob changed during authentication")
    if repository_proposal != canonical_proposal_bytes(proposal):
        raise Item123AuthorizationError("proposal D blob differs from approved bytes")
    if _git_output(
        root,
        ("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"),
    ):
        raise Item123AuthorizationError("trusted workflow checkout has repository drift")


def _json_object(payload: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Item123AuthorizationError(f"{name} is not valid JSON") from error
    if type(value) is not dict:
        raise Item123AuthorizationError(f"{name} must be one JSON object")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise Item123AuthorizationError(f"{name} must be one exact object")
    return value


def _required(value: Mapping[str, Any], key: str, expected_type: type[Any], *, name: str) -> Any:
    item = value.get(key)
    if type(item) is not expected_type:
        raise Item123AuthorizationError(f"{name}.{key} has the wrong exact type")
    return item


def _runtime_distribution_bindings(
    repository_root: Path,
) -> tuple[Item123RuntimeDistributionBinding, Item123NativeRuntimeBinding]:
    payload = _repository_file(
        repository_root,
        ITEM12_3_RUNTIME_DISTRIBUTION_LOCK_RELATIVE_PATH,
        name="runtime distribution lock",
    )
    root = _json_object(payload, name="runtime distribution lock")
    if (
        root.get("runtime_distribution_lock_schema")
        != "cellstate-sciplex3-v5-runtime-distribution-lock"
        or root.get("runtime_distribution_lock_version") != "1.0.0"
    ):
        raise Item123AuthorizationError("runtime distribution lock schema is not exact")
    archive = _mapping(root.get("archive"), name="runtime distribution archive")
    distribution = _mapping(root.get("distribution"), name="runtime distribution locator")
    daemon = _mapping(root.get("runtime_daemon"), name="runtime distribution daemon")
    store_status = daemon.get("image_store_driver_status")
    if store_status != ["driver-type", "io.containerd.snapshotter.v1"]:
        raise Item123AuthorizationError("runtime distribution image store is not exact")
    distribution_binding = Item123RuntimeDistributionBinding(
        distribution_lock_sha256=_sha256(payload),
        repository=_required(distribution, "repository", str, name="distribution"),
        release_tag=_required(distribution, "release_tag", str, name="distribution"),
        release_target_commit=_required(
            distribution, "release_target_commit", str, name="distribution"
        ),
        release_target_role="immutable-runtime-dependency-provenance-not-execution-code",
        release_target_equals_execution_commit_required=False,
        asset_name=_required(distribution, "asset_name", str, name="distribution"),
        asset_sha256=_required(archive, "sha256", str, name="archive"),
        asset_byte_count=_required(archive, "byte_count", int, name="archive"),
        qualified_asset_url=(
            "https://github.com/"
            f"{_required(distribution, 'repository', str, name='distribution')}"
            "/releases/download/"
            f"{_required(distribution, 'release_tag', str, name='distribution')}/"
            f"{_required(distribution, 'asset_name', str, name='distribution')}"
        ),
        attestation_predicate_type="https://in-toto.io/attestation/release/v0.2",
        oci_index_digest=_required(archive, "oci_index_digest", str, name="archive"),
        image_digest=_required(archive, "image_digest", str, name="archive"),
        config_digest=_required(archive, "config_digest", str, name="archive"),
        runtime_image_lock_sha256=_required(
            root, "runtime_image_lock_sha256", str, name="distribution lock"
        ),
        provider=_required(distribution, "provider", str, name="distribution"),
        immutable_release_required=_required(
            distribution, "immutable_release_required", bool, name="distribution"
        ),
        release_attestation_required=_required(
            distribution, "release_attestation_required", bool, name="distribution"
        ),
        asset_attestation_required=_required(
            distribution, "asset_attestation_required", bool, name="distribution"
        ),
    )
    native_binding = Item123NativeRuntimeBinding(
        runner_operating_system="linux",
        runner_architecture="x86_64",
        host_operating_system=_required(
            daemon, "host_operating_system", str, name="runtime daemon"
        ),
        host_architecture=_required(daemon, "host_architecture", str, name="runtime daemon"),
        docker_server_version=_required(daemon, "server_version", str, name="runtime daemon"),
        docker_operating_system=_required(daemon, "operating_system", str, name="runtime daemon"),
        docker_architecture=_required(daemon, "architecture", str, name="runtime daemon"),
        image_store_status=("driver-type", "io.containerd.snapshotter.v1"),
        cgroup_version=_required(daemon, "cgroup_version", str, name="runtime daemon"),
        memory_limit_supported=_required(
            daemon, "memory_limit_supported", bool, name="runtime daemon"
        ),
        memory_swap_limit_supported=_required(
            daemon, "memory_swap_limit_supported", bool, name="runtime daemon"
        ),
        pids_limit_supported=_required(daemon, "pids_limit_supported", bool, name="runtime daemon"),
        local_unix_socket_required=_required(
            daemon, "local_unix_socket_required", bool, name="runtime daemon"
        ),
        emulation_permitted=False,
    )
    return distribution_binding, native_binding


def _validate_runtime_preparation_observation(
    proposal: Item123ExecutionProposal, payload: bytes
) -> str:
    if not payload or len(payload) > 64 * 1024:
        raise ValueError("runtime preparation observation violates its size bound")
    value = _json_object(payload, name="runtime preparation observation")
    if canonical_json_bytes(value) != payload:
        raise ValueError("runtime preparation observation is not canonical JSON")
    archive = _mapping(value.get("archive"), name="runtime preparation archive")
    archive_file = _mapping(value.get("archive_file"), name="runtime preparation archive file")
    release = _mapping(value.get("release"), name="runtime preparation release")
    daemon = _mapping(value.get("daemon"), name="runtime preparation daemon")
    expected_layers = [
        layer.model_dump(mode="json") for layer in proposal.contained_execution.runtime_layers
    ]
    native = proposal.native_runtime
    distribution = proposal.runtime_distribution
    if (
        value.get("distribution_lock_sha256") != distribution.distribution_lock_sha256
        or value.get("runtime_image_lock_sha256") != distribution.runtime_image_lock_sha256
        or value.get("loaded_image_verified") is not True
        or value.get("load_performed_from_verified_descriptor") is not True
        or archive.get("archive_sha256") != distribution.asset_sha256
        or archive.get("index_digest") != distribution.oci_index_digest
        or archive.get("image_digest") != distribution.image_digest
        or archive.get("config_digest") != distribution.config_digest
        or archive.get("layers") != expected_layers
        or archive_file.get("byte_count") != distribution.asset_byte_count
        or archive_file.get("mode") != 0o400
        or release.get("repository") != distribution.repository
        or release.get("release_tag") != distribution.release_tag
        or release.get("release_target_commit") != distribution.release_target_commit
        or release.get("asset_name") != distribution.asset_name
        or release.get("asset_sha256") != distribution.asset_sha256
        or release.get("asset_byte_count") != distribution.asset_byte_count
        or release.get("asset_url") != distribution.qualified_asset_url
        or release.get("attestation_predicate_type") != distribution.attestation_predicate_type
        or release.get("release_attestation_verified") is not True
        or release.get("asset_attestation_verified") is not True
        or daemon.get("host_operating_system") != native.host_operating_system
        or daemon.get("host_architecture") != native.host_architecture
        or daemon.get("server_version") != native.docker_server_version
        or daemon.get("operating_system") != native.docker_operating_system
        or daemon.get("architecture") != native.docker_architecture
        or daemon.get("image_store_status") != list(native.image_store_status)
        or daemon.get("cgroup_version") != native.cgroup_version
        or daemon.get("memory_limit_supported") is not True
        or daemon.get("memory_swap_limit_supported") is not True
        or daemon.get("pids_limit_supported") is not True
    ):
        raise ValueError("runtime preparation observation differs from the approved closure")
    for field in ("device", "inode", "modification_time_ns", "change_time_ns"):
        observed = archive_file.get(field)
        if type(observed) is not int or observed < 0:
            raise ValueError("runtime preparation archive identity is malformed")
    endpoint = daemon.get("endpoint")
    context_name = daemon.get("context_name")
    if (
        type(endpoint) is not str
        or not endpoint.startswith("unix://")
        or type(context_name) is not str
        or not context_name
    ):
        raise ValueError("runtime preparation daemon endpoint is not local and exact")
    return _sha256(payload)


def build_pending_proposal(
    repository_root: Path,
    *,
    execution_repository_commit: str,
    not_before_utc: str,
    expires_at_utc: str,
) -> Item123ExecutionProposal:
    """Reconstruct the complete pending proposal without resolving protected source."""

    commit = _exact_commit(execution_repository_commit, name="execution repository commit")
    policy, code_closure, input_closure, image_lock = contained_training_contracts(
        Path(repository_root)
    )
    distribution, native_runtime = _runtime_distribution_bindings(repository_root)
    source_verification_payload = _repository_file(
        repository_root,
        ITEM12_3_SOURCE_VERIFICATION_RELATIVE_PATH,
        name="source verification descriptor",
    )
    source_verification = _json_object(
        source_verification_payload, name="source verification descriptor"
    )
    source_locator = _mapping(source_verification.get("source"), name="source verification locator")
    if (
        source_locator.get("uri") != ITEM12_3_SOURCE_URI
        or source_locator.get("accession") != "10.5281/zenodo.13350497"
        or source_locator.get("filename") != SCIPLEX3_SOURCE_FILENAME
        or source_locator.get("byte_count") != SCIPLEX3_SOURCE_BYTE_COUNT
        or source_locator.get("sha256") != SCIPLEX3_SOURCE_SHA256
        or source_locator.get("md5") != SCIPLEX3_SOURCE_MD5
    ):
        raise Item123AuthorizationError("public source verification locator is not exact")
    tools = tuple(
        Item123ToolBinding(
            relative_path=relative_path,
            sha256=_sha256(
                _repository_file(repository_root, relative_path, name="authorization tool")
            ),
        )
        for relative_path in sorted(
            (
                ITEM12_3_AUTHORIZATION_CLI_RELATIVE_PATH,
                ITEM12_3_AUTHORIZATION_MODULE_RELATIVE_PATH,
                ITEM12_3_AUTHORIZED_ENTRYPOINT_RELATIVE_PATH,
                ITEM12_3_RUNTIME_DISTRIBUTION_VERIFIER_RELATIVE_PATH,
                ITEM12_3_RUNTIME_OCI_VERIFIER_RELATIVE_PATH,
                ITEM12_3_WORKFLOW_RELATIVE_PATH,
            )
        )
    )
    watchdog_seconds = int(policy.worker_command[2])
    proposal = Item123ExecutionProposal(
        status="pending-explicit-user-approval",
        grants_execution_authority=False,
        approval_mechanism=(
            "workflow-dispatch-exact-proposal-sha256-plus-immutable-one-use-release"
        ),
        repository=ITEM12_3_REPOSITORY,
        repository_owner=ITEM12_3_REPOSITORY_OWNER,
        authorized_dispatch_actor=ITEM12_3_AUTHORIZED_ACTOR,
        workflow_dispatch_ref=ITEM12_3_WORKFLOW_DISPATCH_REF,
        execution_repository_commit=commit,
        not_before_utc=not_before_utc,
        expires_at_utc=expires_at_utc,
        proposal_only_commit_required=True,
        candidate=Item123CandidateBinding(
            candidate_id=SCIPLEX3_CANDIDATE_MODEL_ID,
            implementation_version=SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION,
            model_schema=SCIPLEX3_CANDIDATE_MODEL_SCHEMA,
            model_schema_version=SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION,
            specification_sha256=SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256,
            output_model_schema_sha256=SCIPLEX3_CANDIDATE_OUTPUT_MODEL_SCHEMA_SHA256,
            factor_count=SCIPLEX3_CANDIDATE_FACTOR_COUNT,
            action_count=SCIPLEX3_CANDIDATE_ACTION_COUNT,
            plate_count=SCIPLEX3_CANDIDATE_PLATE_COUNT,
            training_record_count=SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT,
            training_well_count=SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,
            batch_size=SCIPLEX3_CANDIDATE_BATCH_SIZE,
            optimization_seed=0,
            fixed_factor_shape_hex="0x1.999999999999ap-4",
            minimum_outer_iterations=SCIPLEX3_CANDIDATE_MIN_OUTER_ITERATIONS,
            maximum_outer_iterations=SCIPLEX3_CANDIDATE_MAX_OUTER_ITERATIONS,
        ),
        protected_source=Item123ProtectedSourceBinding(
            dataset_id="sciplex3-k562-24h",
            filename=SCIPLEX3_SOURCE_FILENAME,
            public_locator_uri=ITEM12_3_SOURCE_URI,
            accession="10.5281/zenodo.13350497",
            sha256=SCIPLEX3_SOURCE_SHA256,
            md5=SCIPLEX3_SOURCE_MD5,
            byte_count=SCIPLEX3_SOURCE_BYTE_COUNT,
            source_verification_relative_path=ITEM12_3_SOURCE_VERIFICATION_RELATIVE_PATH,
            source_verification_sha256=_sha256(source_verification_payload),
            access_purpose="train_parameters",
            partition_id="p1-train",
            partition_role="train",
            permitted_partitions=("p1-train",),
            prohibited_partitions=(
                "p2-calibration",
                "p3-model-selection-validation",
                "p4-untouched-test",
            ),
            locator_included=True,
            local_path_included=False,
            source_path_caller_supplied=False,
            physical_asset_contains_all_partitions=True,
            entire_opaque_asset_transfer_and_snapshot_required=True,
            permitted_expression_or_raw_count_decode_scope="p1-train-rows-only",
            full_axis_selector_metadata_decode_required=True,
            heldout_selector_metadata_resolved_and_decoded=True,
            heldout_expression_or_raw_count_values_read=False,
            heldout_expression_or_raw_count_values_decoded=False,
            heldout_endpoints_or_outcomes_resolved=False,
            heldout_rows_selected_for_training=False,
            heldout_rows_scored=False,
            heldout_rows_emitted=False,
        ),
        contained_execution=Item123ContainedExecutionBinding(
            policy_fingerprint=policy.fingerprint,
            training_code_closure_sha256=code_closure.fingerprint,
            execution_input_closure_sha256=input_closure.fingerprint,
            runtime_image_lock_fingerprint=image_lock.fingerprint,
            runtime_image_lock_file_sha256=image_lock.image_provenance_sha256,
            runtime_image_reference=image_lock.runtime_image.reference,
            runtime_image_digest=image_lock.runtime_image.digest,
            runtime_archive_sha256=image_lock.archive_sha256,
            runtime_oci_index_digest=image_lock.oci_index_digest,
            runtime_config_digest=image_lock.config_digest,
            runtime_layers=tuple(
                Item123RuntimeLayerBinding(
                    media_type=layer.media_type,
                    digest=layer.digest,
                    byte_count=layer.byte_count,
                )
                for layer in image_lock.layers
            ),
            worker_command=policy.worker_command,
        ),
        runtime_distribution=distribution,
        native_runtime=native_runtime,
        resource_limits=Item123ResourceLimits(
            parent_wall_clock_seconds=policy.wall_clock_seconds,
            protected_source_watchdog_seconds=watchdog_seconds,
            cleanup_timeout_seconds=policy.cleanup_timeout_seconds,
            memory_max_bytes=policy.memory_max_bytes,
            memory_swap_max_bytes=policy.memory_swap_max_bytes,
            pids_limit=policy.pids_limit,
            temporary_max_bytes=policy.temporary_max_bytes,
            snapshot_max_bytes=policy.snapshot_max_bytes,
            workflow_timeout_minutes=150,
            runtime_asset_connect_timeout_seconds=10,
            runtime_asset_total_timeout_seconds=300,
            runtime_asset_retry_count=0,
            source_download_connect_timeout_seconds=30,
            source_download_total_timeout_seconds=1800,
            source_download_retry_count=0,
            protected_source_cleanup_bounded_by_workflow_timeout=True,
            terminal_fallback_timeout_seconds=60,
        ),
        execution_paths=Item123ExecutionPaths(
            execution_id=ITEM12_3_EXECUTION_ID,
            execution_root="/var/lib/cellstate/item12-3",
            staging_root="/var/lib/cellstate/item12-3/staging",
            attempt_ledger_root="/var/lib/cellstate/item12-3/attempt-ledger",
            runtime_archive_path=(
                "/var/lib/cellstate/item12-3/runtime/"
                "sciplex3-v5-runtime-linux-amd64-"
                "37c2fa5846acfbd8357476859bd7f8f0ac6591261d79c2f6f46f0aa22fb76454.oci.tar"
            ),
            protected_source_path=(
                "/var/lib/cellstate/item12-3/protected-source/SrivatsanTrapnell2020_sciplex3.h5ad"
            ),
            canonical_publication_root=(
                "backends/vertical-a/sciplex3-k562-24h-v1/candidate-publication"
            ),
            permitted_stage_relative_paths=_EXPECTED_STAGE_PATHS,
            parent_terminal_report_relative_path=("contained-training-terminal-observation.json"),
            parent_terminal_report_max_bytes=65_536,
            sanitized_terminal_report_filename="item12-3-terminal-report.json",
            sanitized_terminal_report_max_bytes=4_096,
            sanitized_terminal_artifact_retention_days=90,
            durable_terminal_scope=("generic-unknown-state-fallback-in-immutable-attempt-release"),
            actual_terminal_outcome_durability=(
                "actions-artifact-90-days-then-exact-reviewed-repository-commit"
            ),
            authorized_post_run_terminal_repository_path=(
                "audits/item12_3/sciplex3-k562-v5-terminal.json"
            ),
            source_acquisition_started_marker=(
                "/var/lib/cellstate/item12-3/source-acquisition-started.json"
            ),
            source_acquisition_completed_marker=(
                "/var/lib/cellstate/item12-3/source-acquisition-completed.json"
            ),
            host_source_removed_marker=("/var/lib/cellstate/item12-3/host-source-removed.json"),
        ),
        prohibitions=Item123Prohibitions(
            source_persistence_permitted=False,
            source_upload_permitted=False,
            staged_output_upload_permitted=False,
            model_upload_permitted=False,
            canonical_publication_permitted=False,
            runtime_builder_permitted=False,
            p2_p3_p4_access_permitted=False,
            evaluation_or_calibration_permitted=False,
            lifecycle_evidence_permitted=False,
            scientific_admission_permitted=False,
            retry_permitted=False,
            resume_permitted=False,
            only_sanitized_terminal_report_may_leave_runner=True,
            stage_and_model_destroyed_with_ephemeral_runner_after_terminalization=True,
            exact_sanitized_terminal_repository_persistence_permitted=True,
            terminal_persistence_grants_lifecycle_or_scientific_authority=False,
        ),
        stop_policy=Item123StopPolicy(
            maximum_attempt_count=1,
            consume_before_protected_source_acquisition=True,
            stop_after_authorization_failure=True,
            stop_after_preflight_failure=True,
            stop_after_runtime_failure=True,
            stop_after_successful_stage=True,
            no_automatic_retry=True,
            no_manual_reuse=True,
            hard_runner_death_may_leave_only_immutable_generic_fallback=True,
        ),
        authorization_tools=tools,
        authorized_entrypoint=ITEM12_3_AUTHORIZED_ENTRYPOINT_RELATIVE_PATH,
    )
    if policy.training_code_closure_sha256 != code_closure.fingerprint:
        raise Item123AuthorizationError("policy and training code closure are not joined")
    if policy.execution_input_closure_sha256 != input_closure.fingerprint:
        raise Item123AuthorizationError("policy and execution input closure are not joined")
    if image_lock.training_code_closure_sha256 != code_closure.fingerprint:
        raise Item123AuthorizationError("runtime image and training code closure are not joined")
    return proposal


def canonical_proposal_bytes(proposal: Item123ExecutionProposal) -> bytes:
    """Return the exact bytes whose SHA-256 is entered at workflow dispatch."""

    return canonical_json_bytes(proposal.model_dump(mode="json"))


def load_approved_proposal(
    proposal_path: Path,
    approved_proposal_sha256: str,
    *,
    dispatch_actor: str,
    triggering_actor: str,
    github_run_id: int,
    github_run_attempt: int,
    workflow_dispatch_ref: str,
    workflow_repository_commit: str,
    proposal_repository_commit: str,
) -> tuple[Item123ExecutionProposal, Item123DispatchApproval]:
    """Compare raw bytes to user input before parsing or touching any other path."""

    approved = _exact_sha256(approved_proposal_sha256, name="approved proposal digest")
    payload = _read_stable_regular_file(
        Path(proposal_path), name="Item 12.3 proposal", maximum_bytes=ITEM12_3_MAX_PROPOSAL_BYTES
    )
    if _sha256(payload) != approved:
        raise Item123AuthorizationError("checked-in proposal differs from the approved digest")
    try:
        proposal = Item123ExecutionProposal.model_validate_json(payload)
    except ValueError as error:
        raise Item123AuthorizationError(
            "approved proposal violates the Item 12.3 schema"
        ) from error
    if canonical_proposal_bytes(proposal) != payload or proposal.fingerprint != approved:
        raise Item123AuthorizationError("approved proposal is not exact canonical JSON")
    if (
        dispatch_actor != proposal.authorized_dispatch_actor
        or triggering_actor != proposal.authorized_dispatch_actor
        or workflow_dispatch_ref != proposal.workflow_dispatch_ref
    ):
        raise Item123AuthorizationError("workflow actor or trusted dispatch ref is not authorized")
    if github_run_attempt != 1:
        raise Item123AuthorizationError("workflow rerun attempts are not authorized")
    workflow_commit = _exact_commit(workflow_repository_commit, name="workflow repository commit")
    proposal_commit = _exact_commit(proposal_repository_commit, name="proposal repository commit")
    if workflow_commit != proposal.execution_repository_commit:
        raise Item123AuthorizationError("workflow commit differs from approved execution C")
    if proposal_commit == workflow_commit:
        raise Item123AuthorizationError("proposal D must be distinct from workflow/execution C")
    return proposal, Item123DispatchApproval(
        approved_proposal_sha256=approved,
        dispatch_actor=dispatch_actor,
        triggering_actor=triggering_actor,
        github_run_id=github_run_id,
        github_run_attempt=1,
        workflow_dispatch_ref=workflow_dispatch_ref,
        workflow_repository_commit=workflow_commit,
        proposal_repository_commit=proposal_commit,
    )


def require_proposal_current(
    proposal: Item123ExecutionProposal,
    *,
    current_time_utc: datetime | None = None,
) -> None:
    """Reject a not-yet-valid or expired proposal before durable attempt consumption."""

    current = datetime.now(UTC) if current_time_utc is None else current_time_utc
    if current.tzinfo is None or current.utcoffset() != UTC.utcoffset(current):
        raise Item123AuthorizationError("proposal current time must be timezone-aware UTC")
    current = current.astimezone(UTC)
    not_before = _parse_rfc3339_utc(proposal.not_before_utc, name="proposal not-before")
    expires_at = _parse_rfc3339_utc(proposal.expires_at_utc, name="proposal expiry")
    if current < not_before:
        raise Item123AuthorizationError("approved proposal is not yet valid")
    if current >= expires_at:
        raise Item123AuthorizationError("approved proposal has expired")


def verify_proposal_against_repository(
    proposal_path: Path,
    approved_proposal_sha256: str,
    repository_root: Path,
    *,
    dispatch_actor: str,
    triggering_actor: str,
    github_run_id: int,
    github_run_attempt: int,
    workflow_dispatch_ref: str,
    workflow_repository_commit: str,
    proposal_repository_commit: str,
    current_time_utc: datetime | None = None,
) -> tuple[Item123ExecutionProposal, Item123DispatchApproval]:
    """Reconstruct every public binding after the raw approved-digest check succeeds."""

    proposal, approval = load_approved_proposal(
        proposal_path,
        approved_proposal_sha256,
        dispatch_actor=dispatch_actor,
        triggering_actor=triggering_actor,
        github_run_id=github_run_id,
        github_run_attempt=github_run_attempt,
        workflow_dispatch_ref=workflow_dispatch_ref,
        workflow_repository_commit=workflow_repository_commit,
        proposal_repository_commit=proposal_repository_commit,
    )
    require_proposal_current(proposal, current_time_utc=current_time_utc)
    _verify_clean_execution_checkout(
        repository_root, expected_commit=proposal.execution_repository_commit
    )
    _verify_proposal_only_commit(
        repository_root,
        proposal,
        proposal_repository_commit=proposal_repository_commit,
    )
    try:
        expected = build_pending_proposal(
            repository_root,
            execution_repository_commit=proposal.execution_repository_commit,
            not_before_utc=proposal.not_before_utc,
            expires_at_utc=proposal.expires_at_utc,
        )
    except (OSError, TypeError, ValueError) as error:
        raise Item123AuthorizationError(
            "cannot reconstruct the exact source-free proposal"
        ) from error
    if proposal != expected:
        raise Item123AuthorizationError(
            "approved proposal differs from current exact public bindings"
        )
    return proposal, approval


def durable_attempt_tag(proposal_sha256: str) -> str:
    """Return the sole durable attempt name; approval actor and run ID cannot mint another key."""

    return f"item12-3-attempt-{_exact_sha256(proposal_sha256, name='proposal digest')}"


def durable_release_body(
    proposal: Item123ExecutionProposal,
    approval: Item123DispatchApproval,
    *,
    proposal_repository_commit: str,
) -> bytes:
    """Bind trusted workflow C and the data-only proposal D targeted by the release."""

    proposal_commit = _exact_commit(proposal_repository_commit, name="proposal repository commit")
    if proposal_commit != approval.proposal_repository_commit:
        raise Item123AuthorizationError("release body proposal commit differs from approval")
    if (
        approval.approved_proposal_sha256 != proposal.fingerprint
        or approval.workflow_repository_commit != proposal.execution_repository_commit
        or approval.workflow_dispatch_ref != proposal.workflow_dispatch_ref
    ):
        raise Item123AuthorizationError("release body approval or workflow C differs from proposal")
    fallback = build_durable_fallback_terminal(
        proposal,
        approval,
        protected_source_acquisition="unknown_after_global_consumption",
        host_source_cleanup_disposition="unknown",
    )
    fallback_payload = canonical_json_bytes(fallback.model_dump(mode="json"))
    return canonical_json_bytes(
        {
            "approval_fingerprint": approval.fingerprint,
            "approved_proposal_sha256": approval.approved_proposal_sha256,
            "artifact_schema": "cellstate-item12-3-durable-consumption-release",
            "artifact_schema_version": "1.1.0",
            "dispatch_actor": approval.dispatch_actor,
            "execution_id": proposal.execution_paths.execution_id,
            "execution_repository_commit": proposal.execution_repository_commit,
            "fallback_terminal": fallback.model_dump(mode="json"),
            "fallback_terminal_sha256": _sha256(fallback_payload),
            "github_run_attempt": approval.github_run_attempt,
            "github_run_id": approval.github_run_id,
            "no_retry": True,
            "proposal_fingerprint": proposal.fingerprint,
            "proposal_repository_commit": proposal_commit,
            "runtime_distribution_lock_sha256": (
                proposal.runtime_distribution.distribution_lock_sha256
            ),
            "runtime_preparation_required_after_consumption": True,
            "triggering_actor": approval.triggering_actor,
            "workflow_dispatch_ref": approval.workflow_dispatch_ref,
            "workflow_repository_commit": approval.workflow_repository_commit,
        }
    )


def verify_durable_release(
    proposal: Item123ExecutionProposal,
    approval: Item123DispatchApproval,
    *,
    proposal_repository_commit: str,
    release_payload: Mapping[str, Any],
    tag_ref_payload: Mapping[str, Any],
    gh_release_verify_succeeded: bool,
    release_creation_succeeded_in_this_run: bool,
) -> Item123DurableReleaseObservation:
    """Authenticate exact immutable release and lightweight tag readbacks."""

    proposal_commit = _exact_commit(proposal_repository_commit, name="proposal repository commit")
    expected_tag = durable_attempt_tag(approval.approved_proposal_sha256)
    expected_body = durable_release_body(
        proposal, approval, proposal_repository_commit=proposal_commit
    ).decode("utf-8")
    fallback = build_durable_fallback_terminal(
        proposal,
        approval,
        protected_source_acquisition="unknown_after_global_consumption",
        host_source_cleanup_disposition="unknown",
    )
    fallback_sha256 = _sha256(canonical_json_bytes(fallback.model_dump(mode="json")))
    assets = release_payload.get("assets")
    ref_object = tag_ref_payload.get("object")
    if type(assets) is not list or type(ref_object) is not dict:
        raise Item123AuthorizationError("durable release readback is malformed")
    release_id = release_payload.get("id")
    if type(release_id) is not int or isinstance(release_id, bool) or release_id <= 0:
        raise Item123AuthorizationError("durable release ID is malformed")
    if (
        release_payload.get("tag_name") != expected_tag
        or release_payload.get("target_commitish") != proposal_commit
        or release_payload.get("name") != expected_tag
        or release_payload.get("body") != expected_body
        or release_payload.get("draft") is not False
        or release_payload.get("prerelease") is not False
        or release_payload.get("immutable") is not True
        or assets
        or tag_ref_payload.get("ref") != f"refs/tags/{expected_tag}"
        or ref_object.get("type") != "commit"
        or ref_object.get("sha") != proposal_commit
        or gh_release_verify_succeeded is not True
        or release_creation_succeeded_in_this_run is not True
    ):
        raise Item123AuthorizationError("durable release does not consume the exact attempt")
    return Item123DurableReleaseObservation(
        repository=proposal.repository,
        dispatch_actor=approval.dispatch_actor,
        triggering_actor=approval.triggering_actor,
        github_run_id=approval.github_run_id,
        github_run_attempt=approval.github_run_attempt,
        workflow_dispatch_ref=approval.workflow_dispatch_ref,
        tag_name=expected_tag,
        workflow_repository_commit=approval.workflow_repository_commit,
        proposal_repository_commit=proposal_commit,
        proposal_sha256=approval.approved_proposal_sha256,
        proposal_fingerprint=proposal.fingerprint,
        approval_fingerprint=approval.fingerprint,
        execution_repository_commit=proposal.execution_repository_commit,
        execution_id=proposal.execution_paths.execution_id,
        fallback_terminal_sha256=fallback_sha256,
        release_id=release_id,
        release_is_draft=False,
        release_is_prerelease=False,
        release_is_immutable=True,
        release_asset_count=0,
        tag_target_type="commit",
        gh_release_verify_succeeded=True,
        release_creation_succeeded_in_this_run=True,
        no_retry=True,
    )


def attempt_key(
    proposal: Item123ExecutionProposal,
    approval: Item123DispatchApproval,
) -> str:
    """Return the same-host ledger key; the durable release tag remains proposal-only."""

    return canonical_fingerprint(
        {
            "approval_fingerprint": approval.fingerprint,
            "execution_id": proposal.execution_paths.execution_id,
            "proposal_fingerprint": proposal.fingerprint,
        }
    )


def _inspect_ledger_root(path: Path) -> tuple[Path, int]:
    root = Path(os.path.abspath(path))  # noqa: PTH100 - lexical, no resolution
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current /= part
        try:
            state = current.lstat()
        except OSError as error:
            raise Item123AuthorizationError("attempt ledger root is unavailable") from error
        if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
            raise Item123AuthorizationError("attempt ledger path contains a non-directory link")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root, flags)
    except OSError as error:
        raise Item123AuthorizationError("cannot open attempt ledger root") from error
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or stat.S_IMODE(opened.st_mode) != 0o700
        or opened.st_uid != os.geteuid()
        or opened.st_gid != os.getegid()
    ):
        os.close(descriptor)
        raise Item123AuthorizationError("attempt ledger root must be owned by the runner at 0700")
    return root, descriptor


def _lock_ledger(root_descriptor: int) -> int:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(".item12-3.lock", flags, 0o600, dir_fd=root_descriptor)
        os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or opened.st_gid != os.getegid()
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            raise Item123AuthorizationError("attempt ledger lock is not one private regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return descriptor
    except Item123AuthorizationError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise Item123AuthorizationError("cannot acquire attempt ledger lock") from error


def _entry_exists(root_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=root_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise Item123AuthorizationError("cannot inspect attempt ledger entry") from error
    return True


def _write_exclusive_at(
    root_descriptor: int, name: str, payload: bytes, *, maximum_bytes: int
) -> None:
    if not payload or len(payload) > maximum_bytes or "/" in name or name in {".", ".."}:
        raise Item123AuthorizationError("attempt ledger payload or name is invalid")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o400, dir_fd=root_descriptor)
    except FileExistsError as error:
        raise Item123ReplayError("attempt is already or partially consumed") from error
    except OSError as error:
        raise Item123AuthorizationError("cannot create attempt ledger entry") from error
    try:
        os.fchmod(descriptor, 0o400)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise Item123AuthorizationError("attempt ledger entry write was incomplete")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.fsync(root_descriptor)
    except OSError as error:
        # The exclusive marker remains authoritative after any ambiguous durability failure.
        raise Item123ReplayError(
            "attempt consumption durability is ambiguous; reuse is forbidden"
        ) from error


def consume_attempt(
    proposal: Item123ExecutionProposal,
    approval: Item123DispatchApproval,
    durable: Item123DurableReleaseObservation,
    *,
    ledger_root: Path,
) -> tuple[Path, Item123AttemptConsumption]:
    """Write the local immutable marker after, and only after, durable global consumption."""

    if (
        durable.proposal_sha256 != approval.approved_proposal_sha256
        or durable.proposal_fingerprint != proposal.fingerprint
        or durable.approval_fingerprint != approval.fingerprint
        or durable.execution_repository_commit != proposal.execution_repository_commit
        or durable.execution_id != proposal.execution_paths.execution_id
        or durable.tag_name != durable_attempt_tag(approval.approved_proposal_sha256)
    ):
        raise Item123AuthorizationError("durable consumption differs from this exact attempt")
    expected_root = Path(proposal.execution_paths.attempt_ledger_root)
    supplied_root = Path(os.path.abspath(ledger_root))  # noqa: PTH100
    if supplied_root != expected_root:
        raise Item123AuthorizationError("attempt ledger root differs from the approved proposal")
    key = attempt_key(proposal, approval)
    consumption = Item123AttemptConsumption(
        attempt_key=key,
        proposal_sha256=approval.approved_proposal_sha256,
        proposal_fingerprint=proposal.fingerprint,
        approval_fingerprint=approval.fingerprint,
        durable_consumption_fingerprint=durable.fingerprint,
        execution_repository_commit=proposal.execution_repository_commit,
        execution_id=proposal.execution_paths.execution_id,
        protected_source_acquired_at_consumption=False,
        attempt_count=1,
        no_retry=True,
    )
    payload = canonical_json_bytes(consumption.model_dump(mode="json"))
    root, root_descriptor = _inspect_ledger_root(supplied_root)
    lock_descriptor = -1
    try:
        lock_descriptor = _lock_ledger(root_descriptor)
        consumed_name = f"{key}.consumed.json"
        started_name = f"{key}.started.json"
        terminal_name = f"{key}.terminal.json"
        if (
            _entry_exists(root_descriptor, consumed_name)
            or _entry_exists(root_descriptor, started_name)
            or _entry_exists(root_descriptor, terminal_name)
        ):
            raise Item123ReplayError("attempt is already or partially consumed")
        _write_exclusive_at(
            root_descriptor,
            consumed_name,
            payload,
            maximum_bytes=ITEM12_3_MAX_TERMINAL_REPORT_BYTES,
        )
    finally:
        if lock_descriptor >= 0:
            os.close(lock_descriptor)
        os.close(root_descriptor)
    return root / f"{key}.consumed.json", consumption


def _read_canonical_consumption(path: Path) -> Item123AttemptConsumption:
    try:
        state = path.lstat()
    except OSError as error:
        raise Item123AuthorizationError("attempt consumption is unavailable") from error
    if (
        not stat.S_ISREG(state.st_mode)
        or state.st_nlink != 1
        or state.st_uid != os.geteuid()
        or state.st_gid != os.getegid()
        or stat.S_IMODE(state.st_mode) != 0o400
    ):
        raise Item123AuthorizationError("attempt consumption is not one private sealed file")
    payload = _read_stable_regular_file(
        path,
        name="attempt consumption",
        maximum_bytes=ITEM12_3_MAX_TERMINAL_REPORT_BYTES,
    )
    try:
        consumption = Item123AttemptConsumption.model_validate_json(payload)
    except ValueError as error:
        raise Item123AuthorizationError("attempt consumption is malformed") from error
    if canonical_json_bytes(consumption.model_dump(mode="json")) != payload:
        raise Item123AuthorizationError("attempt consumption is not canonical JSON")
    return consumption


def issue_verified_capability(
    proposal: Item123ExecutionProposal,
    approval: Item123DispatchApproval,
    durable: Item123DurableReleaseObservation,
    consumption: Item123AttemptConsumption,
    runtime_preparation_payload: bytes,
) -> VerifiedItem123ExecutionCapability:
    """Join the approved proposal to both durable and local consumption observations."""

    if (
        approval.approved_proposal_sha256 != proposal.fingerprint
        or approval.fingerprint != durable.approval_fingerprint
        or approval.fingerprint != consumption.approval_fingerprint
    ):
        raise Item123AuthorizationError("cannot issue capability from mismatched approval evidence")
    proposal_bytes = canonical_proposal_bytes(proposal)
    runtime_fingerprint = _validate_runtime_preparation_observation(
        proposal, runtime_preparation_payload
    )
    return VerifiedItem123ExecutionCapability(
        capability_state="verified-consumed-one-use",
        proposal=proposal,
        proposal_canonical_json=proposal_bytes.decode("utf-8"),
        proposal_sha256=_sha256(proposal_bytes),
        approval_fingerprint=approval.fingerprint,
        durable_consumption=durable,
        attempt_consumption=consumption,
        runtime_preparation_canonical_json=runtime_preparation_payload.decode("utf-8"),
        runtime_preparation_observation_sha256=runtime_fingerprint,
        protected_source_acquired_at_issue=False,
        retry_permitted=False,
    )


def canonical_capability_bytes(capability: VerifiedItem123ExecutionCapability) -> bytes:
    """Return the bounded canonical receipt passed across checkout of execution commit C."""

    payload = canonical_json_bytes(capability.model_dump(mode="json"))
    if len(payload) > ITEM12_3_MAX_PROPOSAL_BYTES:
        raise Item123AuthorizationError("verified execution capability exceeds its size bound")
    return payload


def load_durable_consumption_observation(path: Path) -> Item123DurableReleaseObservation:
    """Read one canonical durable release observation from a private runner file."""

    payload = _read_stable_regular_file(
        Path(path), name="durable consumption observation", maximum_bytes=64 * 1024
    )
    try:
        observation = Item123DurableReleaseObservation.model_validate_json(payload)
    except ValueError as error:
        raise Item123AuthorizationError("durable consumption observation is malformed") from error
    if canonical_json_bytes(observation.model_dump(mode="json")) != payload:
        raise Item123AuthorizationError("durable consumption observation is not canonical JSON")
    return observation


def load_attempt_consumption(path: Path) -> Item123AttemptConsumption:
    """Public no-follow loader used by the post-runtime capability issuer."""

    return _read_canonical_consumption(Path(path))


def load_runtime_preparation_payload(path: Path, *, proposal: Item123ExecutionProposal) -> bytes:
    """Read and validate the canonical runtime observation captured after consumption."""

    payload = _read_stable_regular_file(
        Path(path), name="runtime preparation observation", maximum_bytes=64 * 1024
    )
    _validate_runtime_preparation_observation(proposal, payload)
    return payload


def load_verified_capability(path: Path) -> VerifiedItem123ExecutionCapability:
    """Read one no-follow canonical capability file without resolving protected source."""

    payload = _read_stable_regular_file(
        Path(path),
        name="verified Item 12.3 execution capability",
        maximum_bytes=ITEM12_3_MAX_PROPOSAL_BYTES,
    )
    try:
        capability = VerifiedItem123ExecutionCapability.model_validate_json(payload)
    except ValueError as error:
        raise Item123AuthorizationError("verified execution capability is malformed") from error
    if canonical_capability_bytes(capability) != payload:
        raise Item123AuthorizationError("verified execution capability is not canonical JSON")
    return capability


def verify_capability_for_execution(
    capability: VerifiedItem123ExecutionCapability,
    *,
    repository_root: Path,
) -> Item123AttemptConsumption:
    """Reconstruct C and re-read its local consumption marker before source-path inspection."""

    if type(capability) is not VerifiedItem123ExecutionCapability:
        raise Item123AuthorizationError("execution requires the exact verified capability type")
    proposal = capability.proposal
    require_proposal_current(proposal)
    _verify_clean_execution_checkout(
        repository_root, expected_commit=proposal.execution_repository_commit
    )
    expected = build_pending_proposal(
        repository_root,
        execution_repository_commit=proposal.execution_repository_commit,
        not_before_utc=proposal.not_before_utc,
        expires_at_utc=proposal.expires_at_utc,
    )
    if proposal != expected:
        raise Item123AuthorizationError(
            "capability proposal differs from execution commit bindings"
        )
    consumption_path = (
        Path(proposal.execution_paths.attempt_ledger_root)
        / f"{capability.attempt_consumption.attempt_key}.consumed.json"
    )
    observed = _read_canonical_consumption(consumption_path)
    if observed != capability.attempt_consumption:
        raise Item123AuthorizationError("capability differs from the sealed local consumption")
    return observed


def claim_execution_start(
    capability: VerifiedItem123ExecutionCapability,
) -> tuple[Path, Item123ExecutionStart]:
    """Atomically consume the capability's only same-run invocation before any source work."""

    if type(capability) is not VerifiedItem123ExecutionCapability:
        raise Item123AuthorizationError("execution start requires the exact capability type")
    proposal = capability.proposal
    consumption = capability.attempt_consumption
    root_path = Path(proposal.execution_paths.attempt_ledger_root)
    consumed_name = f"{consumption.attempt_key}.consumed.json"
    started_name = f"{consumption.attempt_key}.started.json"
    terminal_name = f"{consumption.attempt_key}.terminal.json"
    start = Item123ExecutionStart(
        attempt_key=consumption.attempt_key,
        attempt_consumption_fingerprint=consumption.fingerprint,
        capability_fingerprint=capability.fingerprint,
        proposal_sha256=capability.proposal_sha256,
        github_run_id=capability.durable_consumption.github_run_id,
        github_run_attempt=capability.durable_consumption.github_run_attempt,
        execution_id=proposal.execution_paths.execution_id,
        protected_source_acquired_at_start=False,
        retry_permitted=False,
    )
    payload = canonical_json_bytes(start.model_dump(mode="json"))
    root, root_descriptor = _inspect_ledger_root(root_path)
    lock_descriptor = -1
    try:
        lock_descriptor = _lock_ledger(root_descriptor)
        require_proposal_current(proposal)
        if not _entry_exists(root_descriptor, consumed_name):
            raise Item123AuthorizationError("execution start lost its local consumption")
        if _entry_exists(root_descriptor, started_name) or _entry_exists(
            root_descriptor, terminal_name
        ):
            raise Item123ReplayError("attempt execution already started or terminated")
        observed = _read_canonical_consumption(root / consumed_name)
        if observed != consumption:
            raise Item123AuthorizationError("execution start differs from local consumption")
        _write_exclusive_at(
            root_descriptor,
            started_name,
            payload,
            maximum_bytes=ITEM12_3_MAX_TERMINAL_REPORT_BYTES,
        )
    finally:
        if lock_descriptor >= 0:
            os.close(lock_descriptor)
        os.close(root_descriptor)
    return root / started_name, start


def _read_execution_start(path: Path) -> Item123ExecutionStart:
    payload = _read_stable_regular_file(
        path, name="execution-start receipt", maximum_bytes=ITEM12_3_MAX_TERMINAL_REPORT_BYTES
    )
    try:
        receipt = Item123ExecutionStart.model_validate_json(payload)
    except ValueError as error:
        raise Item123AuthorizationError("execution-start receipt is malformed") from error
    if canonical_json_bytes(receipt.model_dump(mode="json")) != payload:
        raise Item123AuthorizationError("execution-start receipt is not canonical JSON")
    state = path.lstat()
    if (
        state.st_nlink != 1
        or state.st_uid != os.geteuid()
        or state.st_gid != os.getegid()
        or stat.S_IMODE(state.st_mode) != 0o400
    ):
        raise Item123AuthorizationError("execution-start receipt is not private and sealed")
    return receipt


def _read_execution_start_at(root_descriptor: int, name: str) -> Item123ExecutionStart:
    if "/" in name or name in {".", ".."}:
        raise Item123AuthorizationError("execution-start receipt name is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=root_descriptor)
    except OSError as error:
        raise Item123AuthorizationError("execution-start receipt is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or opened.st_gid != os.getegid()
            or stat.S_IMODE(opened.st_mode) != 0o400
            or opened.st_size <= 0
            or opened.st_size > ITEM12_3_MAX_TERMINAL_REPORT_BYTES
        ):
            raise Item123AuthorizationError("execution-start receipt is not private and sealed")
        payload = b""
        while len(payload) < opened.st_size:
            chunk = os.read(descriptor, opened.st_size - len(payload))
            if not chunk:
                raise Item123AuthorizationError("execution-start receipt ended unexpectedly")
            payload += chunk
        if os.read(descriptor, 1):
            raise Item123AuthorizationError("execution-start receipt grew while being read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (opened.st_dev, opened.st_ino, opened.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise Item123AuthorizationError("execution-start receipt changed while being read")
    try:
        receipt = Item123ExecutionStart.model_validate_json(payload)
    except ValueError as error:
        raise Item123AuthorizationError("execution-start receipt is malformed") from error
    if canonical_json_bytes(receipt.model_dump(mode="json")) != payload:
        raise Item123AuthorizationError("execution-start receipt is not canonical JSON")
    return receipt


def verify_execution_start(
    capability: VerifiedItem123ExecutionCapability,
    execution_start: Item123ExecutionStart,
) -> None:
    """Re-read and join the exclusive invocation claim before the supervisor can run."""

    if type(execution_start) is not Item123ExecutionStart:
        raise Item123AuthorizationError("supervisor requires the exact execution-start type")
    ledger_root = Path(capability.proposal.execution_paths.attempt_ledger_root)
    started_name = f"{capability.attempt_consumption.attempt_key}.started.json"
    _, root_descriptor = _inspect_ledger_root(ledger_root)
    lock_descriptor = -1
    try:
        lock_descriptor = _lock_ledger(root_descriptor)
        observed = _read_execution_start_at(root_descriptor, started_name)
    finally:
        if lock_descriptor >= 0:
            os.close(lock_descriptor)
        os.close(root_descriptor)
    if (
        observed != execution_start
        or observed.attempt_key != capability.attempt_consumption.attempt_key
        or observed.attempt_consumption_fingerprint != capability.attempt_consumption.fingerprint
        or observed.capability_fingerprint != capability.fingerprint
        or observed.proposal_sha256 != capability.proposal_sha256
        or observed.github_run_id != capability.durable_consumption.github_run_id
        or observed.github_run_attempt != capability.durable_consumption.github_run_attempt
    ):
        raise Item123AuthorizationError("execution-start receipt differs from the capability")


def build_durable_fallback_terminal(
    proposal: Item123ExecutionProposal,
    approval: Item123DispatchApproval,
    *,
    protected_source_acquisition: Literal[
        "unknown_after_global_consumption", "not_started", "started_incomplete", "completed"
    ],
    host_source_cleanup_disposition: Literal["not_applicable", "removed", "unknown"],
) -> Item123DurableFallbackTerminalReport:
    """Build a pre-release fallback template without inspecting source or runtime paths."""

    unknown = protected_source_acquisition == "unknown_after_global_consumption"
    acquired = protected_source_acquisition != "not_started" if not unknown else False
    report = Item123DurableFallbackTerminalReport(
        proposal_sha256=proposal.fingerprint,
        execution_repository_commit=proposal.execution_repository_commit,
        workflow_repository_commit=approval.workflow_repository_commit,
        proposal_repository_commit=approval.proposal_repository_commit,
        workflow_dispatch_ref=approval.workflow_dispatch_ref,
        attempt_tag=durable_attempt_tag(proposal.fingerprint),
        dispatch_actor=approval.dispatch_actor,
        triggering_actor=approval.triggering_actor,
        github_run_id=approval.github_run_id,
        github_run_attempt=approval.github_run_attempt,
        outcome="runtime_failure",
        reason="terminal_evidence_failed",
        durable_consumption_disposition="creation-attempted-no-retry",
        runtime_preparation_disposition="unavailable",
        protected_source_acquired=None if unknown else acquired,
        protected_source_acquisition=protected_source_acquisition,
        protected_source_cleanup_disposition=(
            "unknown" if acquired or unknown else "not_applicable"
        ),
        host_source_cleanup_disposition=("unknown" if unknown else host_source_cleanup_disposition),
        container_cleanup_disposition=("unknown" if acquired or unknown else "not_started"),
        snapshot_volume_cleanup_disposition=("unknown" if acquired or unknown else "not_started"),
        source_match_disposition=(
            "unknown" if unknown else "acquired_unverified" if acquired else "not_acquired"
        ),
        canonical_publication_performed=False,
        nonterminal_artifact_uploaded=False,
        terminal_report_upload_permitted=True,
        uploaded_payload_scope="terminal-report-only",
        retry_permitted=False,
    )
    payload = canonical_json_bytes(report.model_dump(mode="json"))
    if len(payload) > ITEM12_3_MAX_TERMINAL_REPORT_BYTES:
        raise Item123AuthorizationError("fallback terminal report exceeds its size bound")
    return report


def record_terminal_report(
    consumption_path: Path,
    *,
    outcome: TerminalOutcome,
    reason: TerminalReason,
    protected_source_acquired: bool,
    protected_source_acquisition: Literal[
        "not_started", "started_incomplete", "completed"
    ] = "not_started",
    protected_source_cleanup_disposition: Literal[
        "not_applicable", "removed", "failed", "unknown"
    ] = "not_applicable",
    host_source_cleanup_disposition: Literal[
        "not_applicable", "removed", "failed"
    ] = "not_applicable",
    container_cleanup_disposition: Literal[
        "not_started", "proved_removed", "unproved"
    ] = "not_started",
    snapshot_volume_cleanup_disposition: Literal[
        "not_started", "proved_removed", "unproved"
    ] = "not_started",
    contained_terminal_status: ContainedTerminalStatus = "not_started",
    contained_failure_code: ContainedFailureCode = "not_applicable",
    execution_outcome: Literal[
        "not_started", "unavailable", "success", "timeout", "oom_killed", "worker_failure"
    ] = "not_started",
    parent_wall_clock_elapsed_seconds: float | None = None,
    aggregate_container_limits_enforced: bool = False,
    runtime_preparation_observation_sha256: str | None = None,
    source_match_disposition: Literal[
        "not_acquired",
        "acquired_unverified",
        "pre_and_post_match",
        "pre_match_post_failed",
        "mismatch",
    ] = "not_acquired",
    worker_report_status: Literal[
        "unknown", "not_started", "verified", "missing", "invalid"
    ] = "not_started",
    stage_disposition: Literal["unknown", "not_created", "sealed", "quarantined"] = "not_created",
    stage_semantic_verification: Literal[
        "unknown", "not_attempted", "verified", "failed"
    ] = "not_attempted",
    canonical_publication_unchanged: bool | None = None,
    parent_terminal_observation_sha256: str | None = None,
    staged_success_observation_sha256: str | None = None,
    staged_tree_sha256: str | None = None,
    stage_preserved_on_runner: bool | None = False,
) -> tuple[Path, Item123TerminalReport]:
    """Write one bounded terminal marker without accepting logs, paths, or arbitrary text."""

    path = Path(os.path.abspath(consumption_path))  # noqa: PTH100 - lexical, no resolution
    consumption = _read_canonical_consumption(path)
    expected_name = f"{consumption.attempt_key}.consumed.json"
    if path.name != expected_name:
        raise Item123AuthorizationError("attempt consumption filename is not exact")
    report = Item123TerminalReport(
        attempt_key=consumption.attempt_key,
        attempt_consumption_fingerprint=consumption.fingerprint,
        proposal_sha256=consumption.proposal_sha256,
        execution_id=consumption.execution_id,
        outcome=outcome,
        reason=reason,
        contained_terminal_status=contained_terminal_status,
        contained_failure_code=contained_failure_code,
        execution_outcome=execution_outcome,
        parent_wall_clock_elapsed_seconds=parent_wall_clock_elapsed_seconds,
        aggregate_container_limits_enforced=aggregate_container_limits_enforced,
        runtime_preparation_observation_sha256=runtime_preparation_observation_sha256,
        protected_source_acquired=protected_source_acquired,
        protected_source_acquisition=protected_source_acquisition,
        protected_source_cleanup_disposition=protected_source_cleanup_disposition,
        host_source_cleanup_disposition=host_source_cleanup_disposition,
        container_cleanup_disposition=container_cleanup_disposition,
        snapshot_volume_cleanup_disposition=snapshot_volume_cleanup_disposition,
        source_match_disposition=source_match_disposition,
        worker_report_status=worker_report_status,
        stage_disposition=stage_disposition,
        stage_semantic_verification=stage_semantic_verification,
        canonical_publication_unchanged=canonical_publication_unchanged,
        parent_terminal_observation_sha256=parent_terminal_observation_sha256,
        staged_success_observation_sha256=staged_success_observation_sha256,
        staged_tree_sha256=staged_tree_sha256,
        stage_preserved_on_runner=stage_preserved_on_runner,
        canonical_publication_performed=False,
        nonterminal_artifact_uploaded=False,
        terminal_report_upload_permitted=True,
        uploaded_payload_scope="terminal-report-only",
        retry_permitted=False,
    )
    payload = canonical_json_bytes(report.model_dump(mode="json"))
    if len(payload) > ITEM12_3_MAX_TERMINAL_REPORT_BYTES:
        raise Item123AuthorizationError("terminal report exceeds its exact size bound")
    root, root_descriptor = _inspect_ledger_root(path.parent)
    lock_descriptor = -1
    try:
        lock_descriptor = _lock_ledger(root_descriptor)
        started_name = f"{consumption.attempt_key}.started.json"
        started = _entry_exists(root_descriptor, started_name)
        if not protected_source_acquired and started:
            raise Item123ReplayError(
                "cannot record no-source terminal evidence after execution started"
            )
        if protected_source_acquired:
            if not started:
                raise Item123AuthorizationError(
                    "source-acquired terminal evidence lacks an execution start"
                )
            start = _read_execution_start_at(root_descriptor, started_name)
            if (
                start.attempt_key != consumption.attempt_key
                or start.attempt_consumption_fingerprint != consumption.fingerprint
                or start.proposal_sha256 != consumption.proposal_sha256
            ):
                raise Item123AuthorizationError("terminal evidence differs from execution start")
        if not _entry_exists(root_descriptor, expected_name):
            raise Item123AuthorizationError("attempt consumption disappeared before terminal write")
        terminal_name = f"{consumption.attempt_key}.terminal.json"
        if _entry_exists(root_descriptor, terminal_name):
            raise Item123ReplayError("attempt already has terminal evidence")
        _write_exclusive_at(
            root_descriptor,
            terminal_name,
            payload,
            maximum_bytes=ITEM12_3_MAX_TERMINAL_REPORT_BYTES,
        )
    finally:
        if lock_descriptor >= 0:
            os.close(lock_descriptor)
        os.close(root_descriptor)
    return root / f"{consumption.attempt_key}.terminal.json", report


def record_contained_terminal_report(
    consumption_path: Path,
    contained_terminal_observation: Any,
    *,
    protected_source_acquired: Literal[True],
    runtime_preparation_observation_sha256: str,
    protected_source_cleanup_disposition: Literal["removed", "failed"],
) -> tuple[Path, Item123TerminalReport]:
    """Reduce the typed parent terminal observation to the sole uploadable evidence.

    The execution module imports this module for the capability type, so this reducer avoids a
    reverse import.  It nevertheless requires the exact class module/name and only copies bounded
    enum, Boolean, numeric, and digest fields.
    """

    terminal_type = type(contained_terminal_observation)
    if (
        terminal_type.__module__ != "cellstate.training.execution"
        or terminal_type.__name__ != "ContainedTrainingTerminalObservation"
    ):
        raise Item123AuthorizationError("terminal reduction requires the exact parent type")
    terminal = contained_terminal_observation
    if terminal.protected_source_acquired_before_supervisor is not True:
        raise Item123AuthorizationError("parent terminal lost wrapper-owned acquisition state")
    terminal_status = terminal.terminal_status
    failure_code = terminal.failure_code
    if terminal_status not in {
        "success",
        "timeout",
        "oom_killed",
        "worker_failure",
        "stage_rejected",
        "supervisor_failure",
    } or failure_code not in {
        "none",
        "worker_timed_out",
        "worker_oom_killed",
        "worker_exited_nonzero",
        "worker_report_missing",
        "worker_report_invalid",
        "worker_report_contradiction",
        "stage_inventory_invalid",
        "stage_semantic_verification_failed",
        "canonical_publication_changed",
        "canonical_publication_identity_invalid",
        "contained_executor_failed",
    }:
        raise Item123AuthorizationError("parent terminal status is outside the bounded vocabulary")
    execution = terminal.execution_observation
    raw_execution_outcome = "unavailable" if execution is None else execution.outcome
    if raw_execution_outcome not in {
        "unavailable",
        "success",
        "timeout",
        "oom_killed",
        "worker_failure",
    }:
        raise Item123AuthorizationError(
            "parent execution outcome is outside the bounded vocabulary"
        )
    execution_outcome = cast(
        Literal["unavailable", "success", "timeout", "oom_killed", "worker_failure"],
        raw_execution_outcome,
    )
    elapsed = None if execution is None else execution.parent_wall_clock_elapsed_seconds
    worker = terminal.worker_terminal_report
    if worker is None or worker.source_pre_sha256 is None:
        source_disposition: Literal[
            "not_acquired",
            "acquired_unverified",
            "pre_and_post_match",
            "pre_match_post_failed",
            "mismatch",
        ] = "acquired_unverified"
    elif worker.source_matches_expected:
        source_disposition = "pre_and_post_match"
    elif worker.source_post_sha256 is None:
        source_disposition = "pre_match_post_failed"
    else:
        source_disposition = "mismatch"
    semantic_disposition: Literal["not_attempted", "verified", "failed"] = (
        "verified"
        if terminal.semantic_stage_sha256 is not None
        else "failed"
        if failure_code == "stage_semantic_verification_failed"
        else "not_attempted"
    )
    terminal_payload = canonical_json_bytes(terminal.model_dump(mode="json"))
    contained_success = terminal_status == "success"
    container_cleanup = terminal.container_cleanup_disposition
    snapshot_cleanup = terminal.snapshot_volume_cleanup_disposition
    if container_cleanup not in {"proved_removed", "unproved"} or snapshot_cleanup not in {
        "proved_removed",
        "unproved",
    }:
        raise Item123AuthorizationError("parent terminal cleanup disposition is outside vocabulary")
    overall_cleanup: Literal["removed", "failed", "unknown"]
    if protected_source_cleanup_disposition == "failed":
        overall_cleanup = "failed"
    elif container_cleanup == snapshot_cleanup == "proved_removed":
        overall_cleanup = "removed"
    else:
        overall_cleanup = "unknown"
    success = contained_success and overall_cleanup == "removed"
    staged_success_payload = (
        canonical_json_bytes(terminal.success_observation.model_dump(mode="json"))
        if terminal.success_observation is not None
        else None
    )
    return record_terminal_report(
        consumption_path,
        outcome="success" if success else "runtime_failure",
        reason=(
            "stage_sealed"
            if success
            else "protected_source_cleanup_failed"
            if contained_success
            else "contained_execution_failed"
        ),
        protected_source_acquired=protected_source_acquired,
        protected_source_acquisition="completed",
        protected_source_cleanup_disposition=overall_cleanup,
        host_source_cleanup_disposition=protected_source_cleanup_disposition,
        container_cleanup_disposition=container_cleanup,
        snapshot_volume_cleanup_disposition=snapshot_cleanup,
        contained_terminal_status=terminal_status,
        contained_failure_code=failure_code,
        execution_outcome=execution_outcome,
        parent_wall_clock_elapsed_seconds=elapsed,
        aggregate_container_limits_enforced=terminal.aggregate_container_limits_enforced,
        runtime_preparation_observation_sha256=runtime_preparation_observation_sha256,
        source_match_disposition=source_disposition,
        worker_report_status=terminal.worker_report_status,
        stage_disposition=terminal.stage_disposition,
        stage_semantic_verification=semantic_disposition,
        canonical_publication_unchanged=terminal.canonical_publication_unchanged,
        parent_terminal_observation_sha256=_sha256(terminal_payload),
        staged_success_observation_sha256=(
            _sha256(staged_success_payload) if staged_success_payload is not None else None
        ),
        staged_tree_sha256=terminal.staged_tree_sha256,
        stage_preserved_on_runner=True,
    )


__all__ = [
    "ITEM12_3_AUTHORIZATION_CLI_RELATIVE_PATH",
    "ITEM12_3_AUTHORIZATION_MODULE_RELATIVE_PATH",
    "ITEM12_3_AUTHORIZED_ENTRYPOINT_RELATIVE_PATH",
    "ITEM12_3_LEDGER_ROOT",
    "ITEM12_3_PROPOSAL_RELATIVE_PATH",
    "ITEM12_3_WORKFLOW_DISPATCH_REF",
    "ITEM12_3_WORKFLOW_RELATIVE_PATH",
    "Item123AttemptConsumption",
    "Item123AuthorizationError",
    "Item123DispatchApproval",
    "Item123DurableFallbackTerminalReport",
    "Item123DurableReleaseObservation",
    "Item123ExecutionProposal",
    "Item123ExecutionStart",
    "Item123ReplayError",
    "Item123TerminalReport",
    "VerifiedItem123ExecutionCapability",
    "attempt_key",
    "build_durable_fallback_terminal",
    "build_pending_proposal",
    "canonical_capability_bytes",
    "canonical_proposal_bytes",
    "claim_execution_start",
    "consume_attempt",
    "durable_attempt_tag",
    "durable_release_body",
    "issue_verified_capability",
    "load_approved_proposal",
    "load_attempt_consumption",
    "load_durable_consumption_observation",
    "load_runtime_preparation_payload",
    "load_verified_capability",
    "record_contained_terminal_report",
    "record_terminal_report",
    "require_proposal_current",
    "verify_capability_for_execution",
    "verify_durable_release",
    "verify_execution_start",
    "verify_proposal_against_repository",
]
