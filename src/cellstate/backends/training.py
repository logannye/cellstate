"""Authenticated, p1-only training contracts for a biological component candidate.

This module deliberately stops at ``TRAINED_CANDIDATE``.  A training receipt is evidence, never
authority to open calibration, model-selection, or test partitions and never authority to expose
the public biological runtime.  The exact stage is re-derived from a runtime-only verification
context whose artifact and interface coverage is narrower than full bundle admission.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, BinaryIO, Literal, Protocol, TypeAlias, cast, runtime_checkable

from pydantic import ConfigDict, Field, field_validator, model_validator

from cellstate.data.benchmarks import (
    BenchmarkArtifact,
    BenchmarkPartitionRole,
    ContentAddressedArtifact,
)
from cellstate.data.manifests import SourceArtifact
from cellstate.domain.common import SchemaModel, canonical_fingerprint, canonical_json_bytes
from cellstate.domain.query import StateQuery
from cellstate.training.execution import (
    ContainedExecutionPolicy,
    ContainedTrainingObservation,
    ExecutionInputClosureManifest,
    RuntimeImageLock,
    StagedTrainingInventory,
    TrainingCodeClosureManifest,
)
from cellstate.training.publication import generation_id_for_seed_sha256

from .admission import (
    AdmissionArtifactReference,
    AdmissionVerifierCapability,
    AdmissionVerifierIdentity,
    ArtifactResolutionReceipt,
    ImplementationReceiptRequirement,
    ImplementationReceiptTargetKind,
    LoadedInterfaceReceipt,
    ReceiptAttestation,
    TrustedAdmissionVerifier,
    TrustedRuntimeInterface,
    admission_artifact_reference,
    artifact_coverage_fingerprint,
    attest_canonical_payload,
    implementation_requirement_for_binding,
    require_valid_canonical_attestation,
    require_valid_isolated_loader_observation,
)
from .contracts import (
    BiologicalModelBundleContract,
    BiologicalStagePort,
    BiologicalSupportEnvelope,
    BundleContractKind,
    PortDisposition,
    PortImplementationBinding,
    PortImplementationKind,
    QueryDerivedPrerequisiteReport,
    TrainingRunBinding,
    verify_query_prerequisite_report,
)

TrainingContractSchemaVersion = Literal["0.1-experimental"]
TRAINING_CONTRACT_SCHEMA_VERSION: TrainingContractSchemaVersion = "0.1-experimental"
TRAINED_CANDIDATE_FACTORY_INTERFACE = "cellstate.backends.TrainedCandidateFactory"
_SHA256_PATTERN = r"^[a-fA-F0-9]{64}$"


class TrainingContractModel(SchemaModel):
    """Strict immutable base for the experimental training boundary."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
        strict=True,
    )


def _canonical_text(value: str, *, name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be an exact canonical nonblank string")
    return value


def _canonical_sha256(value: str) -> str:
    if type(value) is not str:
        raise ValueError("SHA-256 values must be exact strings")
    return value.casefold()


def _canonical_values(
    values: tuple[str, ...],
    *,
    name: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise ValueError(f"{name} must be an exact immutable tuple")
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    for value in values:
        _canonical_text(value, name=name)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")
    if values != tuple(sorted(values)):
        raise ValueError(f"{name} must be sorted")
    return values


def _require_utc(value: datetime, *, name: str) -> datetime:
    if type(value) is not datetime:
        raise ValueError(f"{name} must be an exact datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")
    return value.astimezone(UTC)


def _canonical_artifacts(
    artifacts: Iterable[ContentAddressedArtifact],
) -> tuple[ContentAddressedArtifact, ...]:
    by_key: dict[str, ContentAddressedArtifact] = {}
    for artifact in artifacts:
        validated = ContentAddressedArtifact.model_validate(artifact.model_dump(mode="python"))
        previous = by_key.get(validated.artifact_id)
        if previous is not None and previous != validated:
            raise ValueError("one training artifact ID names conflicting byte declarations")
        by_key.setdefault(validated.artifact_id, validated)
    return tuple(by_key[key] for key in sorted(by_key))


def _generation_seed_value(value: object) -> object:
    """Remove only remote locations while retaining every content and semantic identity."""

    if isinstance(value, SchemaModel):
        return _generation_seed_value(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        artifact_keys = {"artifact_id", "uri", "sha256", "byte_count", "media_type"}
        is_content_artifact = artifact_keys <= set(value)
        return {
            str(key): _generation_seed_value(item)
            for key, item in value.items()
            if not (is_content_artifact and key == "uri")
        }
    if isinstance(value, tuple):
        return tuple(_generation_seed_value(item) for item in value)
    if isinstance(value, list):
        return [_generation_seed_value(item) for item in value]
    return value


def candidate_training_plan_generation_seed_bytes(
    plan_or_fields: CandidateTrainingPlan | Mapping[str, object],
) -> bytes:
    """Canonical pre-render plan closure used to choose its immutable generation ID."""

    raw: Mapping[str, Any]
    if isinstance(plan_or_fields, SchemaModel):
        raw = plan_or_fields.model_dump(mode="python")
    elif isinstance(plan_or_fields, Mapping):
        raw = plan_or_fields
    else:
        raise TypeError("candidate training generation seed requires plan fields")
    plan = {
        str(key): value
        for key, value in raw.items()
        if key not in {"planned_generation_id", "publication_generation_seed"}
    }
    return canonical_json_bytes(
        {
            "artifact_schema": "cellstate-candidate-training-generation-seed",
            "artifact_schema_version": "1.0.0",
            "pre_render_plan": _generation_seed_value(plan),
        }
    )


def _canonical_artifact_references(
    artifacts: Iterable[ContentAddressedArtifact | SourceArtifact],
) -> tuple[AdmissionArtifactReference, ...]:
    by_key: dict[str, AdmissionArtifactReference] = {}
    for artifact in artifacts:
        reference = admission_artifact_reference(artifact)
        previous = by_key.get(reference.target_key)
        if previous is not None and previous != reference:
            raise ValueError("one training artifact ID names conflicting byte declarations")
        by_key.setdefault(reference.target_key, reference)
    if not by_key:
        raise ValueError("trained-candidate artifact coverage must not be empty")
    return tuple(by_key[key] for key in sorted(by_key))


def _require_canonical_contract_artifact(
    artifact: ContentAddressedArtifact,
    model: SchemaModel,
    *,
    name: str,
) -> None:
    payload = model.model_dump(mode="json")
    content = canonical_json_bytes(payload)
    if artifact.sha256 != canonical_fingerprint(payload):
        raise ValueError(f"{name} artifact SHA-256 does not match its canonical payload")
    if artifact.byte_count != len(content):
        raise ValueError(f"{name} artifact byte count does not match its canonical payload")
    if artifact.media_type != "application/json":
        raise ValueError(f"{name} artifact must use application/json")


def _require_staged_identity(
    inventory: StagedTrainingInventory,
    *,
    role: str,
    sha256: str,
    byte_count: int,
) -> None:
    entries = tuple(entry for entry in inventory.entries if entry.artifact_role == role)
    if len(entries) != 1 or (entries[0].sha256, entries[0].byte_count) != (
        sha256,
        byte_count,
    ):
        raise ValueError(f"contained stage does not bind the exact {role} bytes")


class CandidateTrainingPlan(TrainingContractModel):
    """Immutable pre-fit scope; it cannot name a future model or final bundle fingerprint."""

    schema_version: TrainingContractSchemaVersion = TRAINING_CONTRACT_SCHEMA_VERSION
    plan_id: str = Field(min_length=1)
    plan_version: str = Field(min_length=1)
    planned_generation_id: str = Field(pattern=_SHA256_PATTERN)
    publication_generation_seed: ContentAddressedArtifact
    query_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    benchmark_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    support_envelope_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    training_partition_ids: tuple[str, ...] = Field(min_length=1)
    training_partition_roles: tuple[BenchmarkPartitionRole, ...] = (BenchmarkPartitionRole.TRAIN,)
    p1_loader_contract: ContentAddressedArtifact
    p1_count_stream: ContentAddressedArtifact
    p1_count_stream_sha256: str = Field(pattern=_SHA256_PATTERN)
    p1_finalized_count_scan_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    p1_assembly_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    p1_design_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    ordered_feature_keys_sha256: str = Field(pattern=_SHA256_PATTERN)
    action_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    target_value_schema_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_specification: ContentAddressedArtifact
    output_model_schema: ContentAddressedArtifact
    runtime_lock: ContentAddressedArtifact
    contained_execution_policy: ContentAddressedArtifact
    runtime_image_lock: ContentAddressedArtifact
    training_code_closure: ContentAddressedArtifact
    training_execution_input_closure: ContentAddressedArtifact
    trainer_implementation: PortImplementationBinding
    candidate_factory_implementation: PortImplementationBinding
    optimization_seed: int = Field(ge=0)
    deterministic_thread_count: int = Field(default=1, ge=1)
    future_calibration_plan: ContentAddressedArtifact | None = None

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))

    @field_validator(
        "query_fingerprint",
        "benchmark_fingerprint",
        "support_envelope_fingerprint",
        "p1_count_stream_sha256",
        "p1_finalized_count_scan_fingerprint",
        "p1_assembly_fingerprint",
        "p1_design_fingerprint",
        "ordered_feature_keys_sha256",
        "action_binding_sha256",
        "target_value_schema_sha256",
        "planned_generation_id",
    )
    @classmethod
    def hashes_are_canonical(cls, value: str) -> str:
        return _canonical_sha256(value)

    @field_validator("optimization_seed", "deterministic_thread_count")
    @classmethod
    def integers_are_exact(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("training plan integer fields must be exact integers")
        return value

    @model_validator(mode="after")
    def scope_is_exact_and_prefit(self) -> CandidateTrainingPlan:
        _canonical_text(self.plan_id, name="training plan ID")
        _canonical_text(self.plan_version, name="training plan version")
        _canonical_values(
            self.training_partition_ids,
            name="training partition IDs",
            allow_empty=False,
        )
        if self.training_partition_roles != (BenchmarkPartitionRole.TRAIN,) or len(
            self.training_partition_roles
        ) != len(self.training_partition_ids):
            raise ValueError("pre-fit plan must bind exactly one typed training partition role")
        artifacts = _canonical_artifacts(
            (
                self.p1_loader_contract,
                self.p1_count_stream,
                self.candidate_specification,
                self.output_model_schema,
                self.runtime_lock,
                self.contained_execution_policy,
                self.runtime_image_lock,
                self.training_code_closure,
                self.training_execution_input_closure,
                self.publication_generation_seed,
                self.trainer_implementation.code_artifact,
                self.candidate_factory_implementation.code_artifact,
            )
        )
        if len(artifacts) != 12:
            raise ValueError("training plan byte roles must have distinct artifact IDs")
        seed_payload = candidate_training_plan_generation_seed_bytes(self)
        if (
            self.publication_generation_seed.media_type != "application/json"
            or self.publication_generation_seed.sha256 != hashlib.sha256(seed_payload).hexdigest()
            or self.publication_generation_seed.byte_count != len(seed_payload)
        ):
            raise ValueError("training plan publication-generation seed is stale")
        if self.planned_generation_id != generation_id_for_seed_sha256(
            self.publication_generation_seed.sha256
        ):
            raise ValueError("training plan generation ID differs from its pre-render seed")
        for implementation, role in (
            (self.trainer_implementation, "trainer"),
            (self.candidate_factory_implementation, "candidate factory"),
        ):
            if implementation.kind is not PortImplementationKind.PYTHON_ENTRY_POINT:
                raise ValueError(f"{role} must declare an exact Python entry point")
        if (
            self.trainer_implementation.entrypoint
            == self.candidate_factory_implementation.entrypoint
            or self.trainer_implementation.code_artifact.sha256
            == self.candidate_factory_implementation.code_artifact.sha256
        ):
            raise ValueError("trainer and candidate factory code identities must be distinct")
        if self.candidate_factory_implementation.interface != TRAINED_CANDIDATE_FACTORY_INTERFACE:
            raise ValueError(
                "candidate factory must bind the canonical trained-candidate interface"
            )
        return self


class P1TrainingEvidence(TrainingContractModel):
    """Typed, closed p1 count-stream evidence; every held-out access flag is false."""

    schema_version: TrainingContractSchemaVersion = TRAINING_CONTRACT_SCHEMA_VERSION
    evidence_id: str = Field(min_length=1)
    evidence_version: str = Field(min_length=1)
    training_plan_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    partition_ids: tuple[str, ...] = Field(min_length=1)
    source: SourceArtifact
    finalized_count_scan: ContentAddressedArtifact
    assembly_receipt: ContentAddressedArtifact
    p1_materialization: ContentAddressedArtifact
    contained_execution_observation: ContentAddressedArtifact
    count_stream_sha256: str = Field(pattern=_SHA256_PATTERN)
    finalized_count_scan_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    assembly_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    record_count: int = Field(gt=0)
    well_count: int = Field(gt=0)
    treated_well_count: int = Field(gt=0)
    control_well_count: int = Field(gt=0)
    nnz: int = Field(gt=0)
    zero_panel_record_count: int = Field(ge=0)
    accessed_partition_roles: tuple[BenchmarkPartitionRole, ...] = (BenchmarkPartitionRole.TRAIN,)
    source_descriptor_reverified: Literal[True] = True
    exact_record_coverage: Literal[True] = True
    source_closed_and_reauthenticated: Literal[True] = True
    p2_membership_read: Literal[False] = False
    p2_cases_read: Literal[False] = False
    p2_outcomes_read: Literal[False] = False
    p3_membership_read: Literal[False] = False
    p3_cases_read: Literal[False] = False
    p3_outcomes_read: Literal[False] = False
    p4_membership_read: Literal[False] = False
    p4_cases_read: Literal[False] = False
    p4_outcomes_read: Literal[False] = False

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))

    @field_validator(
        "training_plan_fingerprint",
        "count_stream_sha256",
        "finalized_count_scan_fingerprint",
        "assembly_fingerprint",
    )
    @classmethod
    def hashes_are_canonical(cls, value: str) -> str:
        return _canonical_sha256(value)

    @field_validator(
        "record_count",
        "well_count",
        "treated_well_count",
        "control_well_count",
        "nnz",
        "zero_panel_record_count",
    )
    @classmethod
    def counts_are_exact(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("p1 training evidence counts must be exact integers")
        return value

    @model_validator(mode="after")
    def evidence_is_p1_only(self) -> P1TrainingEvidence:
        _canonical_text(self.evidence_id, name="p1 training evidence ID")
        _canonical_text(self.evidence_version, name="p1 training evidence version")
        _canonical_values(self.partition_ids, name="p1 partition IDs", allow_empty=False)
        if self.accessed_partition_roles != (BenchmarkPartitionRole.TRAIN,):
            raise ValueError("p1 evidence may access only the training partition role")
        if self.well_count != self.treated_well_count + self.control_well_count:
            raise ValueError("p1 treated/control wells must exactly partition all wells")
        artifacts = _canonical_artifacts(
            (
                self.finalized_count_scan,
                self.assembly_receipt,
                self.p1_materialization,
                self.contained_execution_observation,
            )
        )
        if len(artifacts) != 4:
            raise ValueError("p1 evidence byte roles must have distinct artifact IDs")
        return self


class TrainingSourceSelectionReceipt(TrainingContractModel):
    """Plan-scoped HMAC receipt selecting the only source fit code may open."""

    schema_version: TrainingContractSchemaVersion = TRAINING_CONTRACT_SCHEMA_VERSION
    receipt_kind: Literal["training_source_selection"] = "training_source_selection"
    selection_id: str = Field(min_length=1)
    training_plan_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    workflow_resolution_artifacts: tuple[ContentAddressedArtifact, ...] = Field(min_length=1)
    sources: tuple[SourceArtifact, ...] = Field(min_length=1)
    source_close_reauthentication_required: Literal[True] = True
    selector: AdmissionVerifierIdentity
    selector_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    issued_at: datetime
    attestation: ReceiptAttestation
    selection_fingerprint: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("selection_id")
    @classmethod
    def selection_id_is_canonical(cls, value: str) -> str:
        return _canonical_text(value, name="training-source selection ID")

    @field_validator(
        "training_plan_fingerprint",
        "selector_fingerprint",
        "selection_fingerprint",
    )
    @classmethod
    def hashes_are_canonical(cls, value: str) -> str:
        return _canonical_sha256(value)

    @field_validator("issued_at")
    @classmethod
    def issued_at_is_utc(cls, value: datetime) -> datetime:
        return _require_utc(value, name="training-source selection issuance time")

    @model_validator(mode="after")
    def receipt_is_exact(self) -> TrainingSourceSelectionReceipt:
        workflows = _canonical_artifacts(self.workflow_resolution_artifacts)
        if workflows != self.workflow_resolution_artifacts:
            raise ValueError("workflow artifacts must be sorted by artifact ID")
        source_ids = tuple(source.source_id for source in self.sources)
        _canonical_values(source_ids, name="training source IDs", allow_empty=False)
        if self.selector_fingerprint != self.selector.fingerprint:
            raise ValueError("training-source selector fingerprint does not match its identity")
        if AdmissionVerifierCapability.TRAINING_SOURCE_SELECTION not in self.selector.capabilities:
            raise ValueError("selector lacks training-source selection capability")
        attested = self.model_dump(
            mode="python",
            exclude={"attestation", "selection_fingerprint"},
        )
        if self.attestation.attested_payload_fingerprint != canonical_fingerprint(attested):
            raise ValueError("training-source attestation binds a different selection")
        expected = canonical_fingerprint(
            self.model_dump(mode="python", exclude={"selection_fingerprint"})
        )
        if self.selection_fingerprint != expected:
            raise ValueError("training-source selection fingerprint does not match its payload")
        return self


ObservedTrainingBytes: TypeAlias = bytes | BinaryIO | Iterable[bytes]


def _observe_bytes(source: ObservedTrainingBytes) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0

    def consume(chunk: bytes) -> None:
        nonlocal byte_count
        if type(chunk) is not bytes:
            raise TypeError("training byte streams must yield exact bytes")
        digest.update(chunk)
        byte_count += len(chunk)

    if isinstance(source, bytes):
        consume(source)
    elif hasattr(source, "read"):
        stream = cast(BinaryIO, source)
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            consume(chunk)
    else:
        for chunk in source:
            consume(chunk)
    if byte_count == 0:
        raise ValueError("training byte observations must not be empty")
    return digest.hexdigest(), byte_count


def issue_training_source_selection_receipt(
    *,
    selection_id: str,
    plan: CandidateTrainingPlan,
    workflow_resolution_artifacts: Sequence[ContentAddressedArtifact],
    sources: Sequence[SourceArtifact],
    trusted_selector: TrustedAdmissionVerifier,
    issued_at: datetime,
) -> TrainingSourceSelectionReceipt:
    """Authenticate the trusted workflow's exact p1 source choice before fitting."""

    plan = CandidateTrainingPlan.model_validate(plan.model_dump(mode="python"))
    workflows = _canonical_artifacts(workflow_resolution_artifacts)
    source_tuple = tuple(sorted(sources, key=lambda item: item.source_id))
    payload = {
        "schema_version": TRAINING_CONTRACT_SCHEMA_VERSION,
        "receipt_kind": "training_source_selection",
        "selection_id": selection_id,
        "training_plan_fingerprint": plan.fingerprint,
        "workflow_resolution_artifacts": workflows,
        "sources": source_tuple,
        "source_close_reauthentication_required": True,
        "selector": trusted_selector.identity,
        "selector_fingerprint": trusted_selector.identity.fingerprint,
        "issued_at": issued_at,
    }
    attestation = attest_canonical_payload(
        payload,
        trusted_verifier=trusted_selector,
        required_capability=AdmissionVerifierCapability.TRAINING_SOURCE_SELECTION,
    )
    receipt_payload = {**payload, "attestation": attestation}
    return TrainingSourceSelectionReceipt.model_validate(
        {
            **receipt_payload,
            "selection_fingerprint": canonical_fingerprint(receipt_payload),
        }
    )


def _trusted_root_for(
    *,
    identity: AdmissionVerifierIdentity,
    identity_fingerprint: str,
    attestation: ReceiptAttestation,
    trusted_verifiers: Sequence[TrustedAdmissionVerifier],
) -> TrustedAdmissionVerifier:
    if not trusted_verifiers:
        raise ValueError("training verification requires an external verifier trust root")
    matches = tuple(
        verifier
        for verifier in trusted_verifiers
        if verifier.identity.fingerprint == identity_fingerprint
        and verifier.key_id == attestation.key_id
        and verifier.identity == identity
    )
    if len(matches) != 1:
        raise ValueError("training receipt verifier is absent or ambiguous in the trust root")
    return matches[0]


def require_valid_training_source_selection(
    receipt: TrainingSourceSelectionReceipt,
    *,
    plan: CandidateTrainingPlan,
    trusted_selector: TrustedAdmissionVerifier,
) -> TrainingSourceSelectionReceipt:
    """Re-authenticate a persisted source choice against its pre-fit plan and trust root."""

    receipt = TrainingSourceSelectionReceipt.model_validate(receipt.model_dump(mode="python"))
    plan = CandidateTrainingPlan.model_validate(plan.model_dump(mode="python"))
    if receipt.training_plan_fingerprint != plan.fingerprint:
        raise ValueError("training-source selection is bound to a different pre-fit plan")
    attested = receipt.model_dump(
        mode="python",
        exclude={"attestation", "selection_fingerprint"},
    )
    require_valid_canonical_attestation(
        attested,
        receipt.attestation,
        verifier_identity=receipt.selector,
        trusted_verifier=trusted_selector,
        required_capability=AdmissionVerifierCapability.TRAINING_SOURCE_SELECTION,
    )
    return receipt


class CandidateFitReceipt(TrainingContractModel):
    """HMAC-authenticated semantic verification of one closed, reloadable model artifact."""

    schema_version: TrainingContractSchemaVersion = TRAINING_CONTRACT_SCHEMA_VERSION
    receipt_kind: Literal["candidate_fit"] = "candidate_fit"
    receipt_id: str = Field(min_length=1)
    training_plan_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    source_selection_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    p1_training_evidence_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    contained_execution_observation_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    training_result: ContentAddressedArtifact
    model_artifact: ContentAddressedArtifact
    behavior_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    accessed_partition_roles: tuple[BenchmarkPartitionRole, ...] = (BenchmarkPartitionRole.TRAIN,)
    source_closed_and_reauthenticated: Literal[True] = True
    model_closed_reread_rehashed: Literal[True] = True
    training_complete: Literal[True] = True
    converged: Literal[True] = True
    finite: Literal[True] = True
    behavior_state_recomputed: Literal[True] = True
    candidate_load_verified: Literal[True] = True
    p2_read: Literal[False] = False
    p3_read: Literal[False] = False
    p4_read: Literal[False] = False
    calibration_applied: Literal[False] = False
    model_selection_applied: Literal[False] = False
    public_runtime_authority: Literal[False] = False
    verifier: AdmissionVerifierIdentity
    verifier_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    issued_at: datetime
    attestation: ReceiptAttestation
    receipt_fingerprint: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("receipt_id")
    @classmethod
    def receipt_id_is_canonical(cls, value: str) -> str:
        return _canonical_text(value, name="candidate-fit receipt ID")

    @field_validator(
        "training_plan_fingerprint",
        "source_selection_fingerprint",
        "p1_training_evidence_fingerprint",
        "contained_execution_observation_fingerprint",
        "behavior_manifest_sha256",
        "verifier_fingerprint",
        "receipt_fingerprint",
    )
    @classmethod
    def hashes_are_canonical(cls, value: str) -> str:
        return _canonical_sha256(value)

    @field_validator("issued_at")
    @classmethod
    def issued_at_is_utc(cls, value: datetime) -> datetime:
        return _require_utc(value, name="candidate-fit issuance time")

    @model_validator(mode="after")
    def receipt_is_exact(self) -> CandidateFitReceipt:
        if self.accessed_partition_roles != (BenchmarkPartitionRole.TRAIN,):
            raise ValueError("candidate fitting may access only the training partition role")
        if self.training_result.artifact_id == self.model_artifact.artifact_id:
            raise ValueError("training result and fitted model must be distinct artifacts")
        if self.verifier_fingerprint != self.verifier.fingerprint:
            raise ValueError("candidate-fit verifier fingerprint does not match its identity")
        if AdmissionVerifierCapability.TRAINING_FIT_SEMANTICS not in self.verifier.capabilities:
            raise ValueError("verifier lacks candidate-fit semantic capability")
        attested = self.model_dump(
            mode="python",
            exclude={"attestation", "receipt_fingerprint"},
        )
        if self.attestation.attested_payload_fingerprint != canonical_fingerprint(attested):
            raise ValueError("candidate-fit attestation binds a different result")
        expected = canonical_fingerprint(
            self.model_dump(mode="python", exclude={"receipt_fingerprint"})
        )
        if self.receipt_fingerprint != expected:
            raise ValueError("candidate-fit receipt fingerprint does not match its payload")
        return self


def issue_candidate_fit_receipt(
    *,
    receipt_id: str,
    plan: CandidateTrainingPlan,
    source_selection: TrainingSourceSelectionReceipt,
    p1_evidence: P1TrainingEvidence,
    contained_execution_observation: ContainedTrainingObservation,
    training_result: ContentAddressedArtifact,
    model_artifact: ContentAddressedArtifact,
    observed_model_content: ObservedTrainingBytes,
    behavior_manifest_sha256: str,
    trusted_verifier: TrustedAdmissionVerifier,
    issued_at: datetime,
) -> CandidateFitReceipt:
    """Issue only after rereading the final model bytes and verifying all fit semantics."""

    plan = CandidateTrainingPlan.model_validate(plan.model_dump(mode="python"))
    source_selection = TrainingSourceSelectionReceipt.model_validate(
        source_selection.model_dump(mode="python")
    )
    evidence = P1TrainingEvidence.model_validate(p1_evidence.model_dump(mode="python"))
    execution_observation = ContainedTrainingObservation.model_validate(
        contained_execution_observation.model_dump(mode="python")
    )
    if source_selection.training_plan_fingerprint != plan.fingerprint:
        raise ValueError("candidate fit source selection is bound to another plan")
    if evidence.training_plan_fingerprint != plan.fingerprint:
        raise ValueError("candidate fit evidence is bound to another plan")
    if execution_observation.training_plan_fingerprint != plan.fingerprint:
        raise ValueError("candidate fit execution observation is bound to another plan")
    worker = execution_observation.worker_observation
    if (
        worker.expected_source_sha256 != evidence.source.sha256
        or worker.expected_source_byte_count != evidence.source.byte_count
    ):
        raise ValueError("candidate fit execution observation used another source")
    inventory = execution_observation.staged_inventory
    plan_payload = canonical_json_bytes(plan.model_dump(mode="json"))
    for role, artifact_sha256, artifact_byte_count in (
        ("training_plan", hashlib.sha256(plan_payload).hexdigest(), len(plan_payload)),
        ("training_result", training_result.sha256, training_result.byte_count),
        ("model_artifact", model_artifact.sha256, model_artifact.byte_count),
        (
            "p1_finalized_count_scan",
            evidence.finalized_count_scan.sha256,
            evidence.finalized_count_scan.byte_count,
        ),
        (
            "p1_assembly_receipt",
            evidence.assembly_receipt.sha256,
            evidence.assembly_receipt.byte_count,
        ),
    ):
        _require_staged_identity(
            inventory,
            role=role,
            sha256=artifact_sha256,
            byte_count=artifact_byte_count,
        )
    observed_sha256, observed_byte_count = _observe_bytes(observed_model_content)
    if observed_sha256 != model_artifact.sha256 or observed_byte_count != model_artifact.byte_count:
        raise ValueError("closed model bytes do not match the declared output artifact")
    payload = {
        "schema_version": TRAINING_CONTRACT_SCHEMA_VERSION,
        "receipt_kind": "candidate_fit",
        "receipt_id": receipt_id,
        "training_plan_fingerprint": plan.fingerprint,
        "source_selection_fingerprint": source_selection.selection_fingerprint,
        "p1_training_evidence_fingerprint": evidence.fingerprint,
        "contained_execution_observation_fingerprint": execution_observation.fingerprint,
        "training_result": training_result,
        "model_artifact": model_artifact,
        "behavior_manifest_sha256": behavior_manifest_sha256,
        "accessed_partition_roles": (BenchmarkPartitionRole.TRAIN,),
        "source_closed_and_reauthenticated": True,
        "model_closed_reread_rehashed": True,
        "training_complete": True,
        "converged": True,
        "finite": True,
        "behavior_state_recomputed": True,
        "candidate_load_verified": True,
        "p2_read": False,
        "p3_read": False,
        "p4_read": False,
        "calibration_applied": False,
        "model_selection_applied": False,
        "public_runtime_authority": False,
        "verifier": trusted_verifier.identity,
        "verifier_fingerprint": trusted_verifier.identity.fingerprint,
        "issued_at": issued_at,
    }
    attestation = attest_canonical_payload(
        payload,
        trusted_verifier=trusted_verifier,
        required_capability=AdmissionVerifierCapability.TRAINING_FIT_SEMANTICS,
    )
    receipt_payload = {**payload, "attestation": attestation}
    return CandidateFitReceipt.model_validate(
        {**receipt_payload, "receipt_fingerprint": canonical_fingerprint(receipt_payload)}
    )


def require_valid_candidate_fit_receipt(
    receipt: CandidateFitReceipt,
    *,
    plan: CandidateTrainingPlan,
    source_selection: TrainingSourceSelectionReceipt,
    p1_evidence: P1TrainingEvidence,
    contained_execution_observation: ContainedTrainingObservation,
    trusted_verifier: TrustedAdmissionVerifier,
) -> CandidateFitReceipt:
    """Re-authenticate exact fit semantics; a serialized receipt is never self-trusting."""

    receipt = CandidateFitReceipt.model_validate(receipt.model_dump(mode="python"))
    if receipt.training_plan_fingerprint != plan.fingerprint:
        raise ValueError("candidate-fit receipt is bound to another plan")
    if receipt.source_selection_fingerprint != source_selection.selection_fingerprint:
        raise ValueError("candidate-fit receipt is bound to another source selection")
    if receipt.p1_training_evidence_fingerprint != p1_evidence.fingerprint:
        raise ValueError("candidate-fit receipt is bound to other p1 evidence")
    if (
        receipt.contained_execution_observation_fingerprint
        != contained_execution_observation.fingerprint
    ):
        raise ValueError("candidate-fit receipt is bound to another contained execution")
    attested = receipt.model_dump(
        mode="python",
        exclude={"attestation", "receipt_fingerprint"},
    )
    require_valid_canonical_attestation(
        attested,
        receipt.attestation,
        verifier_identity=receipt.verifier,
        trusted_verifier=trusted_verifier,
        required_capability=AdmissionVerifierCapability.TRAINING_FIT_SEMANTICS,
    )
    return receipt


@runtime_checkable
class TrainedCandidateFactory(Protocol):
    """Exact class entry point used to load and sample one immutable candidate artifact."""

    @classmethod
    def load_exact(cls, model_bytes: bytes, *, expected_sha256: str) -> object: ...

    @property
    def model_artifact_sha256(self) -> str: ...

    def supports(self, request: object) -> bool: ...

    def sample(self, request: object) -> object: ...

    def model_bytes(self) -> bytes: ...

    def behavior_manifest(self) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class TrainingVerificationContext:
    """Runtime-only trust inputs for the p1-to-trained-candidate transition."""

    plan: CandidateTrainingPlan
    plan_artifact: ContentAddressedArtifact
    p1_evidence: P1TrainingEvidence
    p1_evidence_artifact: ContentAddressedArtifact
    contained_execution_policy: ContainedExecutionPolicy
    runtime_image_lock: RuntimeImageLock
    training_code_closure: TrainingCodeClosureManifest
    training_execution_input_closure: ExecutionInputClosureManifest
    contained_execution_observation: ContainedTrainingObservation
    source_selection: TrainingSourceSelectionReceipt
    fit_receipt: CandidateFitReceipt
    query_prerequisite_report: QueryDerivedPrerequisiteReport
    artifact_receipts: tuple[ArtifactResolutionReceipt, ...]
    candidate_factory_interface_receipt: LoadedInterfaceReceipt
    trusted_verifiers: tuple[TrustedAdmissionVerifier, ...]
    runtime_interfaces: Mapping[str, TrustedRuntimeInterface]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan",
            CandidateTrainingPlan.model_validate(self.plan.model_dump(mode="python")),
        )
        for field_name in ("plan_artifact", "p1_evidence_artifact"):
            artifact = cast(ContentAddressedArtifact, getattr(self, field_name))
            object.__setattr__(
                self,
                field_name,
                ContentAddressedArtifact.model_validate(artifact.model_dump(mode="python")),
            )
        object.__setattr__(
            self,
            "p1_evidence",
            P1TrainingEvidence.model_validate(self.p1_evidence.model_dump(mode="python")),
        )
        object.__setattr__(
            self,
            "contained_execution_policy",
            ContainedExecutionPolicy.model_validate(
                self.contained_execution_policy.model_dump(mode="python")
            ),
        )
        object.__setattr__(
            self,
            "runtime_image_lock",
            RuntimeImageLock.model_validate(self.runtime_image_lock.model_dump(mode="python")),
        )
        object.__setattr__(
            self,
            "training_code_closure",
            TrainingCodeClosureManifest.model_validate(
                self.training_code_closure.model_dump(mode="python")
            ),
        )
        object.__setattr__(
            self,
            "training_execution_input_closure",
            ExecutionInputClosureManifest.model_validate(
                self.training_execution_input_closure.model_dump(mode="python")
            ),
        )
        object.__setattr__(
            self,
            "contained_execution_observation",
            ContainedTrainingObservation.model_validate(
                self.contained_execution_observation.model_dump(mode="python")
            ),
        )
        object.__setattr__(
            self,
            "source_selection",
            TrainingSourceSelectionReceipt.model_validate(
                self.source_selection.model_dump(mode="python")
            ),
        )
        object.__setattr__(
            self,
            "fit_receipt",
            CandidateFitReceipt.model_validate(self.fit_receipt.model_dump(mode="python")),
        )
        object.__setattr__(
            self,
            "query_prerequisite_report",
            QueryDerivedPrerequisiteReport.model_validate(
                self.query_prerequisite_report.model_dump(mode="python")
            ),
        )
        object.__setattr__(
            self,
            "artifact_receipts",
            tuple(
                ArtifactResolutionReceipt.model_validate(item.model_dump(mode="python"))
                for item in self.artifact_receipts
            ),
        )
        object.__setattr__(
            self,
            "candidate_factory_interface_receipt",
            LoadedInterfaceReceipt.model_validate(
                self.candidate_factory_interface_receipt.model_dump(mode="python")
            ),
        )
        object.__setattr__(
            self,
            "trusted_verifiers",
            tuple(
                TrustedAdmissionVerifier(
                    identity=AdmissionVerifierIdentity.model_validate(
                        item.identity.model_dump(mode="python")
                    ),
                    key_id=item.key_id,
                    secret=item.secret,
                )
                for item in self.trusted_verifiers
            ),
        )
        runtime_interfaces = {
            name: TrustedRuntimeInterface(
                declared_interface=item.declared_interface,
                interface_artifact=ContentAddressedArtifact.model_validate(
                    item.interface_artifact.model_dump(mode="python")
                ),
                runtime_interface=item.runtime_interface,
            )
            for name, item in self.runtime_interfaces.items()
        }
        verifier_keys = tuple(
            (item.identity.fingerprint, item.key_id) for item in self.trusted_verifiers
        )
        if not verifier_keys or verifier_keys != tuple(sorted(verifier_keys)):
            raise ValueError("training verifier trust roots must be nonempty and sorted")
        if len(verifier_keys) != len(set(verifier_keys)):
            raise ValueError("training verifier trust roots must be unique")
        names = tuple(runtime_interfaces)
        if names != tuple(sorted(names)):
            raise ValueError("training runtime interface registry must be sorted")
        if any(
            name != interface.declared_interface for name, interface in runtime_interfaces.items()
        ):
            raise ValueError("training runtime interface keys must match declarations")
        receipt_keys = tuple(receipt.artifact.target_key for receipt in self.artifact_receipts)
        if receipt_keys != tuple(sorted(receipt_keys)) or len(receipt_keys) != len(
            set(receipt_keys)
        ):
            raise ValueError("training artifact receipts must be unique and sorted")
        object.__setattr__(
            self,
            "runtime_interfaces",
            MappingProxyType(runtime_interfaces),
        )


def training_evidence_artifacts_for_context(
    context: TrainingVerificationContext,
) -> tuple[ContentAddressedArtifact, ...]:
    """Derive the exact content-addressed p1 evidence set stored on TrainingRunBinding."""

    plan = context.plan
    evidence = context.p1_evidence
    fit = context.fit_receipt
    return _canonical_artifacts(
        (
            context.plan_artifact,
            context.p1_evidence_artifact,
            plan.p1_loader_contract,
            plan.p1_count_stream,
            plan.candidate_specification,
            plan.output_model_schema,
            plan.runtime_lock,
            plan.contained_execution_policy,
            plan.runtime_image_lock,
            plan.training_code_closure,
            plan.training_execution_input_closure,
            plan.publication_generation_seed,
            plan.trainer_implementation.code_artifact,
            plan.candidate_factory_implementation.code_artifact,
            evidence.finalized_count_scan,
            evidence.assembly_receipt,
            evidence.p1_materialization,
            evidence.contained_execution_observation,
            fit.training_result,
            *context.source_selection.workflow_resolution_artifacts,
        )
    )


def trained_candidate_required_artifacts(
    bundle: BiologicalModelBundleContract,
    *,
    context: TrainingVerificationContext,
    candidate_interface_requirement: ImplementationReceiptRequirement,
) -> tuple[AdmissionArtifactReference, ...]:
    """Stage-only byte coverage; benchmark descendants and p2/p3/p4 are never expanded."""

    if bundle.model_artifact is None or bundle.training_run is None:
        raise ValueError("trained-candidate artifact coverage requires model and training run")
    content_artifacts = (
        bundle.query.artifact,
        bundle.benchmark.artifact,
        bundle.support_envelope.artifact,
        bundle.model_artifact,
        bundle.training_run.artifact,
        candidate_interface_requirement.interface_artifact,
        *training_evidence_artifacts_for_context(context),
    )
    return _canonical_artifact_references((*content_artifacts, *context.source_selection.sources))


class TrainedCandidateVerification(TrainingContractModel):
    """Serializable audit result of a fresh stage verification; it carries no authority."""

    schema_version: TrainingContractSchemaVersion = TRAINING_CONTRACT_SCHEMA_VERSION
    verification_kind: Literal["trained_candidate"] = "trained_candidate"
    bundle_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    training_run_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    training_plan_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    p1_training_evidence_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    source_selection_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    fit_receipt_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    model_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    required_artifacts: tuple[AdmissionArtifactReference, ...] = Field(min_length=1)
    artifact_coverage_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    candidate_interface_requirement: ImplementationReceiptRequirement
    candidate_interface_receipt_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    verification_fingerprint: str = Field(pattern=_SHA256_PATTERN)

    @field_validator(
        "bundle_fingerprint",
        "training_run_fingerprint",
        "training_plan_fingerprint",
        "p1_training_evidence_fingerprint",
        "source_selection_fingerprint",
        "fit_receipt_fingerprint",
        "model_artifact_sha256",
        "artifact_coverage_fingerprint",
        "candidate_interface_receipt_fingerprint",
        "verification_fingerprint",
    )
    @classmethod
    def hashes_are_canonical(cls, value: str) -> str:
        return _canonical_sha256(value)

    @model_validator(mode="after")
    def verification_is_self_consistent(self) -> TrainedCandidateVerification:
        keys = tuple(item.target_key for item in self.required_artifacts)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("trained-candidate required artifacts must be unique and sorted")
        if self.artifact_coverage_fingerprint != artifact_coverage_fingerprint(
            self.required_artifacts
        ):
            raise ValueError("trained-candidate artifact coverage fingerprint is stale")
        expected = canonical_fingerprint(
            self.model_dump(mode="python", exclude={"verification_fingerprint"})
        )
        if self.verification_fingerprint != expected:
            raise ValueError("trained-candidate verification fingerprint is stale")
        return self


def _require_contract_reference(
    artifact: ContentAddressedArtifact,
    model: SchemaModel,
    *,
    name: str,
) -> None:
    _require_canonical_contract_artifact(artifact, model, name=name)


def _require_artifact_receipt_trust(
    receipt: ArtifactResolutionReceipt,
    *,
    trusted_verifiers: Sequence[TrustedAdmissionVerifier],
) -> None:
    trusted = _trusted_root_for(
        identity=receipt.verifier,
        identity_fingerprint=receipt.verifier_fingerprint,
        attestation=receipt.attestation,
        trusted_verifiers=trusted_verifiers,
    )
    payload = receipt.model_dump(
        mode="python",
        exclude={"attestation", "receipt_fingerprint"},
    )
    require_valid_canonical_attestation(
        payload,
        receipt.attestation,
        verifier_identity=receipt.verifier,
        trusted_verifier=trusted,
        required_capability=AdmissionVerifierCapability.ARTIFACT_BYTE_RESOLUTION,
    )


def _require_interface_receipt_trust(
    receipt: LoadedInterfaceReceipt,
    *,
    trusted_verifiers: Sequence[TrustedAdmissionVerifier],
) -> None:
    trusted = _trusted_root_for(
        identity=receipt.verifier,
        identity_fingerprint=receipt.verifier_fingerprint,
        attestation=receipt.attestation,
        trusted_verifiers=trusted_verifiers,
    )
    payload = receipt.model_dump(
        mode="python",
        exclude={"attestation", "receipt_fingerprint"},
    )
    require_valid_canonical_attestation(
        payload,
        receipt.attestation,
        verifier_identity=receipt.verifier,
        trusted_verifier=trusted,
        required_capability=AdmissionVerifierCapability.LOADED_INTERFACE_VERIFICATION,
    )
    observation = receipt.isolated_loader_observation
    trusted_loader = _trusted_root_for(
        identity=observation.loader,
        identity_fingerprint=observation.loader_fingerprint,
        attestation=observation.attestation,
        trusted_verifiers=trusted_verifiers,
    )
    require_valid_isolated_loader_observation(observation, trusted_loader=trusted_loader)


def require_exact_trained_candidate(
    bundle: BiologicalModelBundleContract,
    *,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    support_envelope: BiologicalSupportEnvelope,
    training_run: TrainingRunBinding,
    context: TrainingVerificationContext,
) -> TrainedCandidateVerification:
    """Derive a trusted candidate from exact p1 evidence without opening held-out surfaces."""

    bundle = BiologicalModelBundleContract.model_validate(bundle.model_dump(mode="python"))
    training_run = TrainingRunBinding.model_validate(training_run.model_dump(mode="python"))
    plan = CandidateTrainingPlan.model_validate(context.plan.model_dump(mode="python"))
    evidence = P1TrainingEvidence.model_validate(context.p1_evidence.model_dump(mode="python"))
    selection = TrainingSourceSelectionReceipt.model_validate(
        context.source_selection.model_dump(mode="python")
    )
    fit = CandidateFitReceipt.model_validate(context.fit_receipt.model_dump(mode="python"))

    if bundle.bundle_kind is not BundleContractKind.COMPONENT_MODEL:
        raise ValueError("only a component-model declaration can become a trained candidate")
    if bundle.model_artifact is None or bundle.training_run is None:
        raise ValueError("trained candidate requires an exact model and training-run reference")
    _require_contract_reference(bundle.training_run.artifact, training_run, name="training run")
    if bundle.training_run.contract_id != training_run.run_id or (
        bundle.training_run.contract_version != training_run.run_version
    ):
        raise ValueError("bundle training-run identity does not match the supplied run")
    if training_run.model_artifact != bundle.model_artifact:
        raise ValueError("training run and bundle name different fitted models")
    if (
        training_run.query_fingerprint != query.fingerprint
        or training_run.benchmark_fingerprint != benchmark.fingerprint
        or training_run.support_envelope_fingerprint != support_envelope.fingerprint
    ):
        raise ValueError("training run is bound to a different query/benchmark/support context")
    if (
        plan.query_fingerprint != query.fingerprint
        or plan.benchmark_fingerprint != benchmark.fingerprint
        or plan.support_envelope_fingerprint != support_envelope.fingerprint
    ):
        raise ValueError("pre-fit plan is bound to a different query/benchmark/support context")
    split_plan = benchmark.definition.split_plan
    if split_plan is None:
        raise ValueError("trained-candidate verification requires a benchmark split plan")
    benchmark_training_partition_ids = tuple(
        partition.partition_id
        for partition in split_plan.partitions
        if partition.role is BenchmarkPartitionRole.TRAIN
    )
    if (
        plan.training_partition_ids != benchmark_training_partition_ids
        or evidence.partition_ids != benchmark_training_partition_ids
        or training_run.training_partition_ids != benchmark_training_partition_ids
    ):
        raise ValueError(
            "training plan, evidence, and run must bind the benchmark's exact training partitions"
        )
    if training_run.training_partition_ids != plan.training_partition_ids:
        raise ValueError("training run partitions differ from the pre-fit plan")
    if any(
        (
            training_run.calibration_partition_ids,
            training_run.model_selection_validation_partition_ids,
            training_run.calibration_evidence_artifacts,
            training_run.model_selection_evidence_artifacts,
            training_run.model_selection_freeze_artifact is not None,
        )
    ):
        raise ValueError("trained-candidate verification cannot carry later-stage evidence")
    if evidence.training_plan_fingerprint != plan.fingerprint:
        raise ValueError("p1 evidence is bound to another training plan")
    if evidence.partition_ids != plan.training_partition_ids:
        raise ValueError("p1 evidence partitions differ from the pre-fit plan")
    if (
        evidence.count_stream_sha256 != plan.p1_count_stream_sha256
        or evidence.finalized_count_scan_fingerprint != plan.p1_finalized_count_scan_fingerprint
        or evidence.assembly_fingerprint != plan.p1_assembly_fingerprint
    ):
        raise ValueError("p1 evidence differs from the plan's exact count-stream closure")
    if selection.sources != (evidence.source,):
        raise ValueError("training source selection must name only the exact p1 evidence source")

    _require_canonical_contract_artifact(context.plan_artifact, plan, name="training plan")
    _require_canonical_contract_artifact(
        context.p1_evidence_artifact,
        evidence,
        name="p1 training evidence",
    )
    policy = ContainedExecutionPolicy.model_validate(
        context.contained_execution_policy.model_dump(mode="python")
    )
    image_lock = RuntimeImageLock.model_validate(
        context.runtime_image_lock.model_dump(mode="python")
    )
    code_closure = TrainingCodeClosureManifest.model_validate(
        context.training_code_closure.model_dump(mode="python")
    )
    input_closure = ExecutionInputClosureManifest.model_validate(
        context.training_execution_input_closure.model_dump(mode="python")
    )
    execution_observation = ContainedTrainingObservation.model_validate(
        context.contained_execution_observation.model_dump(mode="python")
    )
    _require_canonical_contract_artifact(
        plan.contained_execution_policy,
        policy,
        name="contained execution policy",
    )
    _require_canonical_contract_artifact(
        plan.runtime_image_lock,
        image_lock,
        name="runtime image lock",
    )
    _require_canonical_contract_artifact(
        plan.training_code_closure,
        code_closure,
        name="training code closure",
    )
    _require_canonical_contract_artifact(
        plan.training_execution_input_closure,
        input_closure,
        name="training execution-input closure",
    )
    _require_canonical_contract_artifact(
        evidence.contained_execution_observation,
        execution_observation,
        name="contained execution observation",
    )
    if (
        image_lock.runtime_image != policy.runtime_image
        or image_lock.runtime_entrypoint != policy.runtime_entrypoint
        or image_lock.container_user_mode != policy.container_user_mode
        or image_lock.snapshot_volume_initialization != policy.snapshot_volume_initialization
        or code_closure.fingerprint != plan.training_code_closure.sha256
        or input_closure.training_code_closure_sha256 != code_closure.fingerprint
        or input_closure.fingerprint != plan.training_execution_input_closure.sha256
        or policy.training_code_closure_sha256 != code_closure.fingerprint
        or policy.execution_input_closure_sha256 != input_closure.fingerprint
        or image_lock.training_code_closure_sha256 != code_closure.fingerprint
        or execution_observation.training_plan_fingerprint != plan.fingerprint
        or execution_observation.policy_fingerprint != policy.fingerprint
        or execution_observation.runtime_image_digest != image_lock.runtime_image.digest
        or execution_observation.execution_observation.container_user_mode
        != policy.container_user_mode
        or execution_observation.training_code_closure_sha256 != code_closure.fingerprint
        or execution_observation.execution_input_closure_sha256 != input_closure.fingerprint
        or execution_observation.worker_observation.expected_source_sha256 != evidence.source.sha256
        or execution_observation.worker_observation.expected_source_byte_count
        != evidence.source.byte_count
    ):
        raise ValueError("contained execution identities differ from the exact pre-fit plan")
    for role, artifact in (
        ("training_plan", context.plan_artifact),
        ("training_result", fit.training_result),
        ("model_artifact", fit.model_artifact),
        ("p1_finalized_count_scan", evidence.finalized_count_scan),
        ("p1_assembly_receipt", evidence.assembly_receipt),
    ):
        _require_staged_identity(
            execution_observation.staged_inventory,
            role=role,
            sha256=artifact.sha256,
            byte_count=artifact.byte_count,
        )
    expected_training_artifacts = training_evidence_artifacts_for_context(context)
    if training_run.training_evidence_artifacts != expected_training_artifacts:
        raise ValueError("training run does not bind the exact typed p1 evidence closure")

    selector = _trusted_root_for(
        identity=selection.selector,
        identity_fingerprint=selection.selector_fingerprint,
        attestation=selection.attestation,
        trusted_verifiers=context.trusted_verifiers,
    )
    require_valid_training_source_selection(selection, plan=plan, trusted_selector=selector)
    fit_verifier = _trusted_root_for(
        identity=fit.verifier,
        identity_fingerprint=fit.verifier_fingerprint,
        attestation=fit.attestation,
        trusted_verifiers=context.trusted_verifiers,
    )
    require_valid_candidate_fit_receipt(
        fit,
        plan=plan,
        source_selection=selection,
        p1_evidence=evidence,
        contained_execution_observation=execution_observation,
        trusted_verifier=fit_verifier,
    )
    if fit.model_artifact != bundle.model_artifact:
        raise ValueError("fit receipt does not bind the bundle's exact model artifact")

    verify_query_prerequisite_report(
        context.query_prerequisite_report,
        query=query,
        support_envelope=support_envelope,
        bundle=bundle,
    )
    if not context.query_prerequisite_report.structurally_satisfied:
        raise ValueError("exact query-derived candidate prerequisites are not satisfied")

    port_binding = next(
        binding
        for binding in bundle.ports
        if binding.port is BiologicalStagePort.POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL
    )
    if (
        port_binding.disposition is not PortDisposition.PROVIDED
        or port_binding.implementation != plan.candidate_factory_implementation
    ):
        raise ValueError("bundle does not provide the exact planned candidate factory")
    trusted_interface = context.runtime_interfaces.get(
        plan.candidate_factory_implementation.interface
    )
    if trusted_interface is None:
        raise ValueError("canonical trained-candidate factory interface is not registered")
    requirement = implementation_requirement_for_binding(
        bundle,
        target_kind=ImplementationReceiptTargetKind.PORT,
        target_id=BiologicalStagePort.POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL.value,
        implementation=plan.candidate_factory_implementation,
        trusted_runtime_interface=trusted_interface,
    )
    required_factory_members = {
        "behavior_manifest",
        "load_exact",
        "model_artifact_sha256",
        "model_bytes",
        "sample",
        "supports",
    }
    observed_factory_members = {
        signature.split("|", maxsplit=1)[0] for signature in requirement.required_member_signatures
    }
    if not required_factory_members <= observed_factory_members:
        raise ValueError(
            "registered candidate factory lacks exact load/support/sample/artifact-state members"
        )
    interface_receipt = LoadedInterfaceReceipt.model_validate(
        context.candidate_factory_interface_receipt.model_dump(mode="python")
    )
    if interface_receipt.requirement != requirement:
        raise ValueError("candidate factory receipt covers a different exact interface")
    _require_interface_receipt_trust(
        interface_receipt,
        trusted_verifiers=context.trusted_verifiers,
    )

    required_artifacts = trained_candidate_required_artifacts(
        bundle,
        context=context,
        candidate_interface_requirement=requirement,
    )
    actual_by_key = {receipt.artifact.target_key: receipt for receipt in context.artifact_receipts}
    if tuple(actual_by_key) != tuple(item.target_key for item in required_artifacts):
        raise ValueError("stage artifact receipts do not exactly cover trained-candidate bytes")
    for admission_artifact in required_artifacts:
        receipt = actual_by_key[admission_artifact.target_key]
        if receipt.artifact != admission_artifact:
            raise ValueError("stage artifact receipt binds stale or substituted bytes")
        _require_artifact_receipt_trust(
            receipt,
            trusted_verifiers=context.trusted_verifiers,
        )

    payload = {
        "schema_version": TRAINING_CONTRACT_SCHEMA_VERSION,
        "verification_kind": "trained_candidate",
        "bundle_fingerprint": bundle.fingerprint,
        "training_run_fingerprint": training_run.fingerprint,
        "training_plan_fingerprint": plan.fingerprint,
        "p1_training_evidence_fingerprint": evidence.fingerprint,
        "source_selection_fingerprint": selection.selection_fingerprint,
        "fit_receipt_fingerprint": fit.receipt_fingerprint,
        "model_artifact_sha256": bundle.model_artifact.sha256,
        "required_artifacts": required_artifacts,
        "artifact_coverage_fingerprint": artifact_coverage_fingerprint(required_artifacts),
        "candidate_interface_requirement": requirement,
        "candidate_interface_receipt_fingerprint": interface_receipt.receipt_fingerprint,
    }
    return TrainedCandidateVerification.model_validate(
        {**payload, "verification_fingerprint": canonical_fingerprint(payload)}
    )


__all__ = [
    "TRAINED_CANDIDATE_FACTORY_INTERFACE",
    "TRAINING_CONTRACT_SCHEMA_VERSION",
    "CandidateFitReceipt",
    "CandidateTrainingPlan",
    "ObservedTrainingBytes",
    "P1TrainingEvidence",
    "TrainedCandidateFactory",
    "TrainedCandidateVerification",
    "TrainingSourceSelectionReceipt",
    "TrainingVerificationContext",
    "candidate_training_plan_generation_seed_bytes",
    "issue_candidate_fit_receipt",
    "issue_training_source_selection_receipt",
    "require_exact_trained_candidate",
    "require_valid_candidate_fit_receipt",
    "require_valid_training_source_selection",
    "trained_candidate_required_artifacts",
    "training_evidence_artifacts_for_context",
]
