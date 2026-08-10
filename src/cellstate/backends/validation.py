"""Typed, byte-verified validation-result semantics for biological admission.

Validation evidence names and generic artifact references do not establish that an evaluation ran
or that it covered the frozen cases.  This module defines a noncircular result manifest, parses its
exact acquired bytes, binds every supporting result artifact, and emits an externally auditable
receipt.  Passing is derived from the complete typed criterion set for the evidence role.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from types import MappingProxyType
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cellstate.data.benchmarks import BenchmarkArtifact, ContentAddressedArtifact
from cellstate.domain.common import (
    CriterionOutcome,
    SchemaModel,
    canonical_fingerprint,
    canonical_json_bytes,
)
from cellstate.domain.query import StateQuery

from .admission import (
    AdmissionArtifactKind,
    AdmissionReceiptBatchReport,
    AdmissionVerifierCapability,
    AdmissionVerifierIdentity,
    ArtifactDeclaration,
    ArtifactResolutionReceipt,
    ImplementationReceiptTargetKind,
    LoadedInterfaceReceipt,
    ObservedByteSource,
    ReceiptAttestation,
    TrustedAdmissionVerifier,
    TrustedJITLoader,
    TrustedRuntimeInterface,
    VerifiedRuntimeHandle,
    admission_artifact_reference,
    artifact_coverage_fingerprint,
    attest_canonical_payload,
    require_exact_receipt_batch_coverage,
    require_trusted_receipt_verifiers,
    require_valid_canonical_attestation,
    reverify_jit_loaded_interface,
)
from .contracts import (
    BiologicalModelBundleContract,
    BiologicalStagePort,
    BiologicalSupportEnvelope,
    BundleReadiness,
    ModelOperation,
    QueryDerivedPrerequisiteReport,
    TrainingRunBinding,
    ValidationEvidenceBinding,
    ValidationEvidenceKind,
    verify_query_prerequisite_report,
)

ValidationResultSchemaVersion = Literal["0.1-experimental"]
VALIDATION_RESULT_SCHEMA_VERSION: ValidationResultSchemaVersion = "0.1-experimental"
VALIDATION_RESULT_MEDIA_TYPE = "application/vnd.cellstate.validation-result+json"
_SHA256_PATTERN = r"^[a-fA-F0-9]{64}$"


class ValidationResultModel(SchemaModel):
    """Strict base for experimental validation-result artifacts and receipts."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
        strict=True,
    )


def _canonical_text(value: str, *, name: str) -> str:
    if not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be nonblank and trimmed")
    return value


def _canonical_values(
    values: tuple[str, ...],
    *,
    name: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    for value in values:
        _canonical_text(value, name=name)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")
    if values != tuple(sorted(values)):
        raise ValueError(f"{name} must be sorted")
    return values


def _utc(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")
    return value.astimezone(UTC)


VALIDATION_CRITERIA_POLICY_ID = "cellstate.validation-semantic-criteria"
VALIDATION_CRITERIA_POLICY_VERSION = "0.1.0"


class ValidationSemanticCriterion(StrEnum):
    """Minimum semantic checks whose meaning is fixed by the validation evidence role."""

    ACCEPTANCE_POLICY_PASSED = "acceptance_policy_passed"
    AUTHORITATIVE_CASE_COVERAGE = "authoritative_case_coverage"
    CALIBRATION_BEHAVIOR = "calibration_behavior"
    EXACT_PARTITION_ISOLATION = "exact_partition_isolation"
    EVALUATION_COMPLETENESS = "evaluation_completeness"
    IMPLEMENTATION_SCOPE_BINDING = "implementation_scope_binding"
    MODEL_SELECTION_FREEZE = "model_selection_freeze"
    NO_UNTOUCHED_TEST_ACCESS = "no_untouched_test_access"
    OOD_ABSTENTION_BEHAVIOR = "ood_abstention_behavior"
    OPERATION_CONTRACT_CONFORMANCE = "operation_contract_conformance"
    PORT_CONTRACT_CONFORMANCE = "port_contract_conformance"
    RESULT_SCHEMA_CONFORMANCE = "result_schema_conformance"
    SCIENTIFIC_SUPPORT_BOUNDS = "scientific_support_bounds"


_COMMON_CRITERIA = frozenset(
    {
        ValidationSemanticCriterion.AUTHORITATIVE_CASE_COVERAGE,
        ValidationSemanticCriterion.IMPLEMENTATION_SCOPE_BINDING,
        ValidationSemanticCriterion.PORT_CONTRACT_CONFORMANCE,
        ValidationSemanticCriterion.RESULT_SCHEMA_CONFORMANCE,
    }
)

REQUIRED_VALIDATION_CRITERIA: Mapping[
    ValidationEvidenceKind,
    frozenset[ValidationSemanticCriterion],
] = MappingProxyType(
    {
        ValidationEvidenceKind.LOCKED_COMPONENT_EVALUATION: _COMMON_CRITERIA
        | frozenset(
            {
                ValidationSemanticCriterion.ACCEPTANCE_POLICY_PASSED,
                ValidationSemanticCriterion.EVALUATION_COMPLETENESS,
                ValidationSemanticCriterion.EXACT_PARTITION_ISOLATION,
            }
        ),
        ValidationEvidenceKind.MODEL_SELECTION: _COMMON_CRITERIA
        | frozenset(
            {
                ValidationSemanticCriterion.EXACT_PARTITION_ISOLATION,
                ValidationSemanticCriterion.MODEL_SELECTION_FREEZE,
                ValidationSemanticCriterion.NO_UNTOUCHED_TEST_ACCESS,
            }
        ),
        ValidationEvidenceKind.PORT_IMPLEMENTATION_VALIDATION: _COMMON_CRITERIA
        | frozenset(
            {
                ValidationSemanticCriterion.PORT_CONTRACT_CONFORMANCE,
                ValidationSemanticCriterion.SCIENTIFIC_SUPPORT_BOUNDS,
            }
        ),
        ValidationEvidenceKind.RUNTIME_OPERATION_VALIDATION: _COMMON_CRITERIA
        | frozenset(
            {
                ValidationSemanticCriterion.OPERATION_CONTRACT_CONFORMANCE,
                ValidationSemanticCriterion.SCIENTIFIC_SUPPORT_BOUNDS,
            }
        ),
        ValidationEvidenceKind.SUPPORT_OOD_VALIDATION: _COMMON_CRITERIA
        | frozenset(
            {
                ValidationSemanticCriterion.OOD_ABSTENTION_BEHAVIOR,
                ValidationSemanticCriterion.SCIENTIFIC_SUPPORT_BOUNDS,
            }
        ),
        ValidationEvidenceKind.UNCERTAINTY_CALIBRATION: _COMMON_CRITERIA
        | frozenset(
            {
                ValidationSemanticCriterion.CALIBRATION_BEHAVIOR,
                ValidationSemanticCriterion.EXACT_PARTITION_ISOLATION,
            }
        ),
    }
)

VALIDATION_CRITERIA_POLICY_FINGERPRINT = canonical_fingerprint(
    {
        "policy_id": VALIDATION_CRITERIA_POLICY_ID,
        "policy_version": VALIDATION_CRITERIA_POLICY_VERSION,
        "base_criteria": {
            kind.value: sorted(criterion.value for criterion in criteria)
            for kind, criteria in REQUIRED_VALIDATION_CRITERIA.items()
        },
        "conditional_rules": {
            "covered_operations": ValidationSemanticCriterion.OPERATION_CONTRACT_CONFORMANCE.value,
            "uncertainty_calibrator": ValidationSemanticCriterion.CALIBRATION_BEHAVIOR.value,
            "ood_or_strict_support": [
                ValidationSemanticCriterion.OOD_ABSTENTION_BEHAVIOR.value,
                ValidationSemanticCriterion.SCIENTIFIC_SUPPORT_BOUNDS.value,
            ],
            "scientific_decision_gates": (
                ValidationSemanticCriterion.SCIENTIFIC_SUPPORT_BOUNDS.value
            ),
        },
    }
)


def required_validation_criteria(
    *,
    evidence_kind: ValidationEvidenceKind,
    covered_ports: Sequence[BiologicalStagePort],
    covered_operations: Sequence[ModelOperation],
) -> frozenset[ValidationSemanticCriterion]:
    """Derive the closed criterion set from evidence role and exact covered surfaces."""

    required = set(REQUIRED_VALIDATION_CRITERIA[evidence_kind])
    ports = set(covered_ports)
    if covered_operations:
        required.add(ValidationSemanticCriterion.OPERATION_CONTRACT_CONFORMANCE)
    if BiologicalStagePort.UNCERTAINTY_CALIBRATOR in ports:
        required.add(ValidationSemanticCriterion.CALIBRATION_BEHAVIOR)
    if ports.intersection(
        {
            BiologicalStagePort.OOD_DETECTOR,
            BiologicalStagePort.STRICT_SUPPORT_OOD_GATE,
        }
    ):
        required.update(
            {
                ValidationSemanticCriterion.OOD_ABSTENTION_BEHAVIOR,
                ValidationSemanticCriterion.SCIENTIFIC_SUPPORT_BOUNDS,
            }
        )
    if ports.intersection(
        {
            BiologicalStagePort.IDENTIFIABILITY_ANALYZER,
            BiologicalStagePort.SUFFICIENCY_EVALUATOR,
            BiologicalStagePort.VALUE_OF_INFORMATION_ENGINE,
        }
    ):
        required.add(ValidationSemanticCriterion.SCIENTIFIC_SUPPORT_BOUNDS)
    return frozenset(required)


class ValidationCriterionResult(ValidationResultModel):
    """One typed criterion outcome with exact result-artifact support."""

    criterion: ValidationSemanticCriterion
    outcome: CriterionOutcome
    evidence_artifact_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    @field_validator("evidence_artifact_ids", "reasons")
    @classmethod
    def values_are_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_values(values, name="validation criterion values")

    @model_validator(mode="after")
    def outcome_has_honest_evidence(self) -> ValidationCriterionResult:
        if self.outcome is CriterionOutcome.PASSED:
            if not self.evidence_artifact_ids or self.reasons:
                raise ValueError("a passed validation criterion requires evidence and no blockers")
        elif self.outcome is CriterionOutcome.FAILED:
            if not self.evidence_artifact_ids or not self.reasons:
                raise ValueError("a failed validation criterion requires evidence and reasons")
        elif self.evidence_artifact_ids or not self.reasons:
            raise ValueError(
                "not-evaluated or unsupported validation criteria require reasons and no "
                "positive evidence"
            )
        return self


class ValidationResultManifest(ValidationResultModel):
    """Canonical result index produced before its own artifact digest is known."""

    schema_version: ValidationResultSchemaVersion = VALIDATION_RESULT_SCHEMA_VERSION
    criteria_policy_id: Literal["cellstate.validation-semantic-criteria"] = (
        "cellstate.validation-semantic-criteria"
    )
    criteria_policy_version: Literal["0.1.0"] = "0.1.0"
    criteria_policy_fingerprint: str = Field(
        default=VALIDATION_CRITERIA_POLICY_FINGERPRINT,
        pattern=_SHA256_PATTERN,
    )
    result_id: str = Field(min_length=1)
    result_version: str = Field(min_length=1)
    validation_scope_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    evidence_id: str = Field(min_length=1)
    evidence_version: str = Field(min_length=1)
    evidence_kind: ValidationEvidenceKind
    query_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    benchmark_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    support_envelope_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    training_run_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    model_artifact_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    implementation_scope_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    partition_ids: tuple[str, ...] = Field(min_length=1)
    evaluation_case_ids: tuple[str, ...] = Field(min_length=1)
    covered_ports: tuple[BiologicalStagePort, ...] = Field(min_length=1)
    covered_operations: tuple[ModelOperation, ...] = ()
    result_manifest_artifact_id: str = Field(min_length=1)
    supporting_artifacts: tuple[ContentAddressedArtifact, ...] = ()
    criteria: tuple[ValidationCriterionResult, ...] = Field(min_length=1)
    generated_at: datetime

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))

    @property
    def passed(self) -> bool:
        return all(item.outcome is CriterionOutcome.PASSED for item in self.criteria)

    @field_validator(
        "validation_scope_fingerprint",
        "criteria_policy_fingerprint",
        "query_fingerprint",
        "benchmark_fingerprint",
        "support_envelope_fingerprint",
        "training_run_fingerprint",
        "model_artifact_fingerprint",
        "implementation_scope_fingerprint",
    )
    @classmethod
    def fingerprints_are_canonical(cls, value: str) -> str:
        return value.casefold()

    @field_validator("generated_at")
    @classmethod
    def generated_at_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, name="validation-result generation time")

    @model_validator(mode="after")
    def manifest_is_exact_and_closed_world(self) -> ValidationResultManifest:
        if self.criteria_policy_fingerprint != VALIDATION_CRITERIA_POLICY_FINGERPRINT:
            raise ValueError("validation result binds a stale semantic-criteria policy")
        for value, name in (
            (self.result_id, "validation result ID"),
            (self.result_version, "validation result version"),
            (self.evidence_id, "validation evidence ID"),
            (self.evidence_version, "validation evidence version"),
            (self.result_manifest_artifact_id, "validation result-manifest artifact ID"),
        ):
            _canonical_text(value, name=name)
        _canonical_values(
            self.partition_ids, name="validation result partitions", allow_empty=False
        )
        _canonical_values(
            self.evaluation_case_ids,
            name="validation result evaluation cases",
            allow_empty=False,
        )
        port_values = tuple(port.value for port in self.covered_ports)
        _canonical_values(port_values, name="validation result covered ports", allow_empty=False)
        operation_values = tuple(operation.value for operation in self.covered_operations)
        _canonical_values(operation_values, name="validation result covered operations")
        supporting_ids = tuple(item.artifact_id for item in self.supporting_artifacts)
        _canonical_values(supporting_ids, name="validation supporting artifacts")
        if self.result_manifest_artifact_id in set(supporting_ids):
            raise ValueError("a result manifest cannot include its own content-addressed artifact")
        expected_criteria = required_validation_criteria(
            evidence_kind=self.evidence_kind,
            covered_ports=self.covered_ports,
            covered_operations=self.covered_operations,
        )
        criterion_values = tuple(item.criterion.value for item in self.criteria)
        _canonical_values(
            criterion_values,
            name="validation semantic criteria",
            allow_empty=False,
        )
        if {item.criterion for item in self.criteria} != expected_criteria:
            raise ValueError(
                "validation result must cover the exact criteria for its evidence kind"
            )
        cited_ids = {
            artifact_id for result in self.criteria for artifact_id in result.evidence_artifact_ids
        }
        unknown = cited_ids - set(supporting_ids)
        if unknown:
            raise ValueError(
                f"validation criteria cite undeclared supporting artifacts: {sorted(unknown)}"
            )
        if cited_ids != set(supporting_ids):
            raise ValueError("every validation supporting artifact must support a typed criterion")
        return self


def _expected_manifest_fields(evidence: ValidationEvidenceBinding) -> dict[str, object]:
    return {
        "validation_scope_fingerprint": evidence.validation_scope_fingerprint,
        "evidence_id": evidence.evidence_id,
        "evidence_version": evidence.evidence_version,
        "evidence_kind": evidence.evidence_kind,
        "query_fingerprint": evidence.query_fingerprint,
        "benchmark_fingerprint": evidence.benchmark_fingerprint,
        "support_envelope_fingerprint": evidence.support_envelope_fingerprint,
        "training_run_fingerprint": evidence.training_run_fingerprint,
        "model_artifact_fingerprint": evidence.model_artifact_fingerprint,
        "implementation_scope_fingerprint": evidence.implementation_scope_fingerprint,
        "partition_ids": evidence.partition_ids,
        "evaluation_case_ids": evidence.evaluation_case_ids,
        "covered_ports": evidence.covered_ports,
        "covered_operations": evidence.covered_operations,
    }


def _manifest_matches_evidence(
    manifest: ValidationResultManifest,
    evidence: ValidationEvidenceBinding,
) -> bool:
    payload = manifest.model_dump(mode="python")
    return all(
        payload[name] == expected for name, expected in _expected_manifest_fields(evidence).items()
    )


class ValidationResultVerificationReceipt(ValidationResultModel):
    """Authenticated bounded observation from an isolated semantic evaluator.

    The evaluator, not the result manifest, is the authority for criterion outcomes.  It must
    independently compute them from the exact authenticated result/support bytes and frozen scope
    before issuing this receipt.  The receipt is still only an auditable record; admission
    re-authenticates it against an external trust root and the authoritative byte-receipt batch.
    """

    schema_version: ValidationResultSchemaVersion = VALIDATION_RESULT_SCHEMA_VERSION
    receipt_kind: Literal["validation_result_verification"] = "validation_result_verification"
    receipt_id: str = Field(min_length=1)
    bundle_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    evidence_binding_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    validation_scope_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    result_manifest_artifact: ContentAddressedArtifact
    result_manifest: ValidationResultManifest
    artifact_receipts: tuple[ArtifactResolutionReceipt, ...] = Field(min_length=1)
    artifact_coverage_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    evaluation_method: Literal["isolated_semantic_evaluator_v1"] = "isolated_semantic_evaluator_v1"
    evaluator: AdmissionVerifierIdentity
    evaluator_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    isolation_evidence_artifact: ContentAddressedArtifact
    issued_at: datetime
    attestation: ReceiptAttestation
    receipt_fingerprint: str = Field(pattern=_SHA256_PATTERN)

    @property
    def passed(self) -> bool:
        return self.result_manifest.passed

    @field_validator("receipt_id")
    @classmethod
    def receipt_id_is_canonical(cls, value: str) -> str:
        return _canonical_text(value, name="validation-result receipt ID")

    @field_validator(
        "bundle_fingerprint",
        "evidence_binding_fingerprint",
        "validation_scope_fingerprint",
        "artifact_coverage_fingerprint",
        "evaluator_fingerprint",
        "receipt_fingerprint",
    )
    @classmethod
    def fingerprints_are_canonical(cls, value: str) -> str:
        return value.casefold()

    @field_validator("issued_at")
    @classmethod
    def issued_at_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, name="validation-result receipt issuance time")

    @model_validator(mode="after")
    def receipt_is_exact(self) -> ValidationResultVerificationReceipt:
        if self.result_manifest_artifact.media_type != VALIDATION_RESULT_MEDIA_TYPE:
            raise ValueError("validation result manifests require their exact media type")
        if (
            self.result_manifest.result_manifest_artifact_id
            != self.result_manifest_artifact.artifact_id
        ):
            raise ValueError("validation result manifest must name its containing artifact")
        embedded_bytes = canonical_json_bytes(self.result_manifest.model_dump(mode="json"))
        if (
            sha256(embedded_bytes).hexdigest() != self.result_manifest_artifact.sha256
            or len(embedded_bytes) != self.result_manifest_artifact.byte_count
        ):
            raise ValueError(
                "embedded validation result must reproduce its exact content-addressed bytes"
            )
        if self.validation_scope_fingerprint != self.result_manifest.validation_scope_fingerprint:
            raise ValueError("validation receipt scope must match its parsed result manifest")
        receipt_ids = tuple(item.receipt_id for item in self.artifact_receipts)
        _canonical_values(receipt_ids, name="validation artifact receipt IDs", allow_empty=False)
        received_artifacts: dict[str, ContentAddressedArtifact] = {}
        for receipt in self.artifact_receipts:
            if (
                receipt.artifact.artifact_kind
                is not AdmissionArtifactKind.CONTENT_ADDRESSED_ARTIFACT
            ):
                raise ValueError("validation result artifacts must use content-addressed receipts")
            artifact = receipt.artifact.content_addressed_artifact
            assert artifact is not None
            if artifact.artifact_id in received_artifacts:
                raise ValueError("validation result artifacts must have one receipt each")
            received_artifacts[artifact.artifact_id] = artifact
            if receipt.issued_at > self.issued_at:
                raise ValueError("validation artifact receipts cannot postdate result verification")
        expected_artifacts = {
            self.result_manifest_artifact.artifact_id: self.result_manifest_artifact,
            **{item.artifact_id: item for item in self.result_manifest.supporting_artifacts},
        }
        if received_artifacts != expected_artifacts:
            raise ValueError("validation receipt must cover every exact result artifact once")
        expected_coverage = artifact_coverage_fingerprint(
            tuple(
                sorted(
                    (
                        admission_artifact_reference(artifact)
                        for artifact in expected_artifacts.values()
                    ),
                    key=lambda item: item.target_key,
                )
            )
        )
        if self.artifact_coverage_fingerprint != expected_coverage:
            raise ValueError("validation semantic receipt binds stale artifact coverage")
        if self.evaluator_fingerprint != self.evaluator.fingerprint:
            raise ValueError(
                "validation semantic evaluator fingerprint does not match its identity"
            )
        if (
            AdmissionVerifierCapability.VALIDATION_RESULT_SEMANTICS
            not in self.evaluator.capabilities
        ):
            raise ValueError("validation evaluator lacks semantic-result capability")
        if self.isolation_evidence_artifact.artifact_id in expected_artifacts:
            raise ValueError("semantic isolation evidence must be distinct from result artifacts")
        if self.result_manifest.generated_at > self.issued_at:
            raise ValueError("a validation result cannot be generated after its verification")
        attested_payload = self.model_dump(
            mode="python",
            exclude={"attestation", "receipt_fingerprint"},
        )
        if self.attestation.attested_payload_fingerprint != canonical_fingerprint(attested_payload):
            raise ValueError("validation semantic attestation binds a different observation")
        expected_fingerprint = canonical_fingerprint(
            self.model_dump(mode="python", exclude={"receipt_fingerprint"})
        )
        if self.receipt_fingerprint != expected_fingerprint:
            raise ValueError("validation-result receipt fingerprint does not match its payload")
        return self


def issue_validation_result_verification_receipt(
    *,
    receipt_id: str,
    bundle: BiologicalModelBundleContract,
    evidence: ValidationEvidenceBinding,
    result_manifest_artifact_id: str,
    result_manifest_bytes: bytes,
    artifact_receipts: Sequence[ArtifactResolutionReceipt],
    trusted_evaluator: TrustedAdmissionVerifier,
    issued_at: datetime,
    isolation_evidence_artifact: ContentAddressedArtifact,
) -> ValidationResultVerificationReceipt:
    """Attest an isolated evaluator's exact computed semantic observation.

    This helper never interprets a filename or caller Boolean as validation.  It may be called only
    by an application-owned isolated evaluator after that evaluator has independently recomputed
    the complete criterion set from the authenticated bytes and frozen protocol.
    """

    artifacts = {item.artifact_id: item for item in evidence.evidence_artifacts}
    manifest_artifact = artifacts.get(result_manifest_artifact_id)
    if manifest_artifact is None:
        raise ValueError("validation result manifest is absent from the evidence artifacts")
    if manifest_artifact.media_type != VALIDATION_RESULT_MEDIA_TYPE:
        raise ValueError("validation result manifest has the wrong media type")
    if sha256(result_manifest_bytes).hexdigest() != manifest_artifact.sha256:
        raise ValueError("validation result-manifest bytes do not match their declared SHA-256")
    if len(result_manifest_bytes) != manifest_artifact.byte_count:
        raise ValueError("validation result-manifest bytes do not match their declared byte count")
    manifest = ValidationResultManifest.model_validate_json(result_manifest_bytes)
    canonical_manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
    if result_manifest_bytes != canonical_manifest_bytes:
        raise ValueError("validation result-manifest bytes must use canonical JSON encoding")
    if not _manifest_matches_evidence(manifest, evidence):
        raise ValueError("validation result manifest does not match its exact evidence binding")
    if manifest.result_manifest_artifact_id != result_manifest_artifact_id:
        raise ValueError("validation result manifest names a different containing artifact")
    supporting = tuple(
        item
        for item in evidence.evidence_artifacts
        if item.artifact_id != result_manifest_artifact_id
    )
    if manifest.supporting_artifacts != supporting:
        raise ValueError("validation result manifest must bind every exact supporting artifact")
    receipts = tuple(sorted(artifact_receipts, key=lambda item: item.receipt_id))
    coverage = artifact_coverage_fingerprint(
        tuple(
            sorted(
                (
                    admission_artifact_reference(artifact)
                    for artifact in (manifest_artifact, *supporting)
                ),
                key=lambda item: item.target_key,
            )
        )
    )
    payload = {
        "schema_version": VALIDATION_RESULT_SCHEMA_VERSION,
        "receipt_kind": "validation_result_verification",
        "receipt_id": receipt_id,
        "bundle_fingerprint": bundle.fingerprint,
        "evidence_binding_fingerprint": evidence.fingerprint,
        "validation_scope_fingerprint": evidence.validation_scope_fingerprint,
        "result_manifest_artifact": manifest_artifact,
        "result_manifest": manifest,
        "artifact_receipts": receipts,
        "artifact_coverage_fingerprint": coverage,
        "evaluation_method": "isolated_semantic_evaluator_v1",
        "evaluator": trusted_evaluator.identity,
        "evaluator_fingerprint": trusted_evaluator.identity.fingerprint,
        "isolation_evidence_artifact": isolation_evidence_artifact,
        "issued_at": issued_at,
    }
    attestation = attest_canonical_payload(
        payload,
        trusted_verifier=trusted_evaluator,
        required_capability=AdmissionVerifierCapability.VALIDATION_RESULT_SEMANTICS,
    )
    receipt_payload = {**payload, "attestation": attestation}
    return ValidationResultVerificationReceipt.model_validate(
        {
            **receipt_payload,
            "receipt_fingerprint": canonical_fingerprint(receipt_payload),
        }
    )


class ValidationResultReceiptBatch(ValidationResultModel):
    """Complete one-to-one semantic verification for a bundle's validation bindings."""

    schema_version: ValidationResultSchemaVersion = VALIDATION_RESULT_SCHEMA_VERSION
    batch_kind: Literal["complete_validation_result_receipts"] = (
        "complete_validation_result_receipts"
    )
    batch_id: str = Field(min_length=1)
    bundle_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    implementation_scope_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    required_evidence_fingerprints: tuple[str, ...] = Field(min_length=1)
    receipts: tuple[ValidationResultVerificationReceipt, ...] = Field(min_length=1)
    issued_at: datetime
    batch_fingerprint: str = Field(pattern=_SHA256_PATTERN)

    @property
    def all_results_passed(self) -> bool:
        return all(receipt.passed for receipt in self.receipts)

    @field_validator("batch_id")
    @classmethod
    def batch_id_is_canonical(cls, value: str) -> str:
        return _canonical_text(value, name="validation-result batch ID")

    @field_validator(
        "bundle_fingerprint",
        "implementation_scope_fingerprint",
        "batch_fingerprint",
    )
    @classmethod
    def fingerprints_are_canonical(cls, value: str) -> str:
        return value.casefold()

    @field_validator("required_evidence_fingerprints")
    @classmethod
    def evidence_fingerprints_are_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(len(value) != 64 for value in values):
            raise ValueError("validation evidence fingerprints must be SHA-256 values")
        return _canonical_values(
            tuple(value.casefold() for value in values),
            name="validation evidence fingerprints",
            allow_empty=False,
        )

    @field_validator("issued_at")
    @classmethod
    def issued_at_is_utc(cls, value: datetime) -> datetime:
        return _utc(value, name="validation-result batch issuance time")

    @model_validator(mode="after")
    def coverage_is_exact(self) -> ValidationResultReceiptBatch:
        receipt_ids = tuple(item.receipt_id for item in self.receipts)
        _canonical_values(receipt_ids, name="validation-result receipt IDs", allow_empty=False)
        actual_fingerprints = tuple(
            sorted(item.evidence_binding_fingerprint for item in self.receipts)
        )
        if actual_fingerprints != self.required_evidence_fingerprints:
            raise ValueError("validation-result receipts must cover every exact evidence binding")
        if len(actual_fingerprints) != len(set(actual_fingerprints)):
            raise ValueError("one evidence binding may have only one semantic result receipt")
        if any(item.bundle_fingerprint != self.bundle_fingerprint for item in self.receipts):
            raise ValueError("validation-result receipts must bind the batch's exact bundle")
        if any(
            item.result_manifest.implementation_scope_fingerprint
            != self.implementation_scope_fingerprint
            for item in self.receipts
        ):
            raise ValueError("validation-result receipts bind a stale implementation scope")
        if any(item.issued_at > self.issued_at for item in self.receipts):
            raise ValueError("validation-result receipts cannot postdate their batch")
        expected = canonical_fingerprint(
            self.model_dump(mode="python", exclude={"batch_fingerprint"})
        )
        if self.batch_fingerprint != expected:
            raise ValueError("validation-result batch fingerprint does not match its payload")
        return self


def build_validation_result_receipt_batch(
    *,
    batch_id: str,
    bundle: BiologicalModelBundleContract,
    validation_evidence: Sequence[ValidationEvidenceBinding],
    receipts: Sequence[ValidationResultVerificationReceipt],
    issued_at: datetime,
) -> ValidationResultReceiptBatch:
    """Build a closed-world batch over every supplied exact validation binding."""

    evidence = tuple(validation_evidence)
    evidence_fingerprints = tuple(sorted(item.fingerprint for item in evidence))
    if len(evidence_fingerprints) != len(set(evidence_fingerprints)):
        raise ValueError("validation evidence bindings must be unique")
    sorted_receipts = tuple(sorted(receipts, key=lambda item: item.receipt_id))
    payload = {
        "schema_version": VALIDATION_RESULT_SCHEMA_VERSION,
        "batch_kind": "complete_validation_result_receipts",
        "batch_id": batch_id,
        "bundle_fingerprint": bundle.fingerprint,
        "implementation_scope_fingerprint": bundle.implementation_scope_fingerprint,
        "required_evidence_fingerprints": evidence_fingerprints,
        "receipts": sorted_receipts,
        "issued_at": issued_at,
    }
    return ValidationResultReceiptBatch(
        schema_version=VALIDATION_RESULT_SCHEMA_VERSION,
        batch_kind="complete_validation_result_receipts",
        batch_id=batch_id,
        bundle_fingerprint=bundle.fingerprint,
        implementation_scope_fingerprint=bundle.implementation_scope_fingerprint,
        required_evidence_fingerprints=evidence_fingerprints,
        receipts=sorted_receipts,
        issued_at=issued_at,
        batch_fingerprint=canonical_fingerprint(payload),
    )


def require_exact_validation_result_coverage(
    bundle: BiologicalModelBundleContract,
    support_envelope: BiologicalSupportEnvelope,
    validation_evidence: Sequence[ValidationEvidenceBinding],
    batch: ValidationResultReceiptBatch,
    *,
    trusted_verifiers: Sequence[TrustedAdmissionVerifier],
    authoritative_receipt_batch: AdmissionReceiptBatchReport,
) -> ValidationResultReceiptBatch:
    """Atomically rebind semantic results to exact declarations, bytes, and trust roots."""

    bundle = BiologicalModelBundleContract.model_validate(bundle.model_dump(mode="python"))
    support_envelope = BiologicalSupportEnvelope.model_validate(
        support_envelope.model_dump(mode="python")
    )
    evidence = tuple(
        ValidationEvidenceBinding.model_validate(item.model_dump(mode="python"))
        for item in validation_evidence
    )
    batch = ValidationResultReceiptBatch.model_validate(batch.model_dump(mode="python"))

    if batch.bundle_fingerprint != bundle.fingerprint:
        raise ValueError("validation-result batch is bound to a different bundle")
    if batch.implementation_scope_fingerprint != bundle.implementation_scope_fingerprint:
        raise ValueError("validation-result batch is bound to a stale implementation scope")
    evidence_ids = tuple(sorted(item.evidence_id for item in evidence))
    bundle_ids = tuple(sorted(item.contract_id for item in bundle.validation_evidence))
    envelope_ids = tuple(
        sorted(item.evidence_id for item in support_envelope.required_validation_evidence)
    )
    if not evidence_ids or evidence_ids != bundle_ids or evidence_ids != envelope_ids:
        raise ValueError(
            "validation semantics must cover every-and-only bundle/envelope evidence binding"
        )
    evidence_by_id = {item.evidence_id: item for item in evidence}
    for reference in bundle.validation_evidence:
        item = evidence_by_id[reference.contract_id]
        encoded = canonical_json_bytes(item.model_dump(mode="json"))
        if (
            reference.contract_version != item.evidence_version
            or reference.artifact.sha256 != item.fingerprint
            or reference.artifact.byte_count != len(encoded)
        ):
            raise ValueError("bundle validation reference does not bind the exact evidence bytes")
    requirement_by_id = {
        item.evidence_id: item for item in support_envelope.required_validation_evidence
    }
    if any(
        evidence_by_id[evidence_id].evidence_kind
        is not requirement_by_id[evidence_id].evidence_kind
        for evidence_id in evidence_ids
    ):
        raise ValueError("support envelope validation-evidence roles do not match")
    expected_fingerprints = tuple(sorted(item.fingerprint for item in evidence))
    if batch.required_evidence_fingerprints != expected_fingerprints:
        raise ValueError("validation-result batch does not cover the exact evidence bindings")
    evidence_by_fingerprint = {item.fingerprint: item for item in evidence}
    trusted_by_key: dict[tuple[str, str], TrustedAdmissionVerifier] = {}
    for verifier in trusted_verifiers:
        key = (verifier.identity.fingerprint, verifier.key_id)
        if key in trusted_by_key:
            raise ValueError("validation trust-root identity/key pairs must be unique")
        trusted_by_key[key] = verifier
    if not trusted_by_key:
        raise ValueError("validation-result verification requires an external verifier trust root")
    authoritative = require_trusted_receipt_verifiers(
        authoritative_receipt_batch,
        trusted_verifiers=trusted_verifiers,
    )
    if (
        authoritative.bundle_fingerprint != bundle.fingerprint
        or authoritative.implementation_scope_fingerprint != bundle.implementation_scope_fingerprint
    ):
        raise ValueError("authoritative byte receipts bind a different bundle scope")
    authoritative_by_artifact = {
        receipt.artifact.fingerprint: receipt.receipt_fingerprint
        for receipt in authoritative.artifact_receipts
    }
    for receipt in batch.receipts:
        evidence_binding = evidence_by_fingerprint.get(receipt.evidence_binding_fingerprint)
        if evidence_binding is None:
            raise ValueError("validation-result receipt references unknown evidence")
        if receipt.validation_scope_fingerprint != evidence_binding.validation_scope_fingerprint:
            raise ValueError("validation-result receipt binds a stale validation scope")
        trusted_evaluator = trusted_by_key.get(
            (receipt.evaluator_fingerprint, receipt.attestation.key_id)
        )
        if trusted_evaluator is None:
            raise ValueError("validation-result receipt was issued by an untrusted evaluator")
        attested_payload = receipt.model_dump(
            mode="python",
            exclude={"attestation", "receipt_fingerprint"},
        )
        require_valid_canonical_attestation(
            attested_payload,
            receipt.attestation,
            verifier_identity=receipt.evaluator,
            trusted_verifier=trusted_evaluator,
            required_capability=AdmissionVerifierCapability.VALIDATION_RESULT_SEMANTICS,
        )
        if not _manifest_matches_evidence(receipt.result_manifest, evidence_binding):
            raise ValueError("parsed validation result no longer matches its evidence binding")
        expected_artifacts = {
            artifact.artifact_id: artifact for artifact in evidence_binding.evidence_artifacts
        }
        actual_artifacts = {
            receipt.result_manifest_artifact.artifact_id: receipt.result_manifest_artifact,
            **{
                artifact.artifact_id: artifact
                for artifact in receipt.result_manifest.supporting_artifacts
            },
        }
        if actual_artifacts != expected_artifacts:
            raise ValueError("validation-result receipt does not bind every evidence artifact")
        for artifact_receipt in receipt.artifact_receipts:
            expected_receipt = authoritative_by_artifact.get(artifact_receipt.artifact.fingerprint)
            if expected_receipt != artifact_receipt.receipt_fingerprint:
                raise ValueError(
                    "validation semantics must reuse the authoritative exact-byte receipt"
                )
    return batch


def _collect_content_artifacts(value: object) -> tuple[ContentAddressedArtifact, ...]:
    """Walk a typed benchmark/run artifact without treating arbitrary dictionaries as authority."""

    found: list[ContentAddressedArtifact] = []

    def visit(item: object) -> None:
        if isinstance(item, ContentAddressedArtifact):
            found.append(item)
        elif isinstance(item, BaseModel):
            for field_name in type(item).model_fields:
                visit(getattr(item, field_name))
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(found)


def admission_execution_artifacts(
    *,
    benchmark: BenchmarkArtifact,
    training_run: TrainingRunBinding | None,
    validation_evidence: Sequence[ValidationEvidenceBinding],
) -> tuple[ContentAddressedArtifact, ...]:
    """Derive non-source artifacts nested in the exact execution/evaluation declarations.

    Real-data sources are deliberately absent. They come only from the authenticated
    ``ExecutionSourceSelectionReceipt`` in the authoritative admission batch, whose selector is
    responsible for resolving the exact science-plus-permission workflow.
    """

    declarations: list[ArtifactDeclaration] = list(_collect_content_artifacts(benchmark))
    if training_run is not None:
        declarations.extend(_collect_content_artifacts(training_run))
    for evidence in validation_evidence:
        declarations.extend(evidence.evidence_artifacts)

    by_target: dict[str, ContentAddressedArtifact] = {}
    by_identity: dict[str, str] = {}
    for declaration in declarations:
        if not isinstance(declaration, ContentAddressedArtifact):
            raise TypeError("nested admission artifacts must be content-addressed declarations")
        reference = admission_artifact_reference(declaration)
        previous = by_identity.get(reference.target_key)
        if previous is not None and previous != reference.fingerprint:
            raise ValueError("one admission artifact ID names conflicting byte declarations")
        by_identity[reference.target_key] = reference.fingerprint
        by_target.setdefault(reference.target_key, declaration)
    return tuple(by_target[key] for key in sorted(by_target))


@dataclass(frozen=True, slots=True)
class AdmissionVerificationContext:
    """Runtime-only inputs rechecked by admission; never a serialized authorization token."""

    receipt_batch: AdmissionReceiptBatchReport
    validation_result_batch: ValidationResultReceiptBatch | None
    query_prerequisite_report: QueryDerivedPrerequisiteReport
    trusted_verifiers: tuple[TrustedAdmissionVerifier, ...]
    runtime_interfaces: Mapping[str, TrustedRuntimeInterface]
    jit_loaders: Mapping[str, TrustedJITLoader] = field(default_factory=dict)

    def __post_init__(self) -> None:
        verifier_keys = tuple(
            (item.identity.fingerprint, item.key_id) for item in self.trusted_verifiers
        )
        if not verifier_keys:
            raise ValueError("admission verification requires an external verifier trust root")
        if len(verifier_keys) != len(set(verifier_keys)):
            raise ValueError("trusted admission verifier identity/key pairs must be unique")
        if verifier_keys != tuple(sorted(verifier_keys)):
            raise ValueError("trusted admission verifier identity/key pairs must be sorted")
        interface_names = tuple(self.runtime_interfaces)
        if interface_names != tuple(sorted(interface_names)):
            raise ValueError("runtime interface registry must be sorted")
        if any(
            name != interface.declared_interface
            for name, interface in self.runtime_interfaces.items()
        ):
            raise ValueError("runtime interface registry keys must match their declarations")
        loader_fingerprints = tuple(self.jit_loaders)
        if loader_fingerprints != tuple(sorted(loader_fingerprints)):
            raise ValueError("trusted JIT loader registry must be sorted")
        trusted_verifier_keys = set(verifier_keys)
        for fingerprint, loader in self.jit_loaders.items():
            if fingerprint != loader.fingerprint:
                raise ValueError("trusted JIT loader registry key differs from its identity")
            loader_key = (loader.verifier.identity.fingerprint, loader.verifier.key_id)
            if loader_key not in trusted_verifier_keys:
                raise ValueError("trusted JIT loader is absent from the external trust root")
        object.__setattr__(
            self,
            "runtime_interfaces",
            MappingProxyType(dict(self.runtime_interfaces)),
        )
        object.__setattr__(
            self,
            "jit_loaders",
            MappingProxyType(dict(self.jit_loaders)),
        )


class JITCodeProvider(Protocol):
    """Application-owned resolver that reacquires admitted code bytes for the JIT loader."""

    def __call__(
        self,
        receipt: LoadedInterfaceReceipt,
    ) -> ObservedByteSource: ...


@dataclass(frozen=True, slots=True)
class BiologicalExecutionAuthorization:
    """Nonserialized authority containing the exact JIT-checked objects that may be invoked."""

    readiness: BundleReadiness
    operation: ModelOperation | None
    runtime_handles: tuple[VerifiedRuntimeHandle, ...]

    def __post_init__(self) -> None:
        if not self.runtime_handles:
            raise ValueError("biological execution authorization requires runtime handles")
        if self.operation is None:
            if not self.readiness.component_execution_allowed:
                raise ValueError("component authorization requires admitted component readiness")
            if any(
                handle.target_kind is ImplementationReceiptTargetKind.RUNTIME_OPERATION
                for handle in self.runtime_handles
            ):
                raise ValueError("component authorization cannot carry public-operation handles")
        elif not self.readiness.runnable:
            raise ValueError("operation authorization requires admitted runtime readiness")
        elif tuple(
            handle.target_id
            for handle in self.runtime_handles
            if handle.target_kind is ImplementationReceiptTargetKind.RUNTIME_OPERATION
        ) != (self.operation.value,):
            raise ValueError("operation authorization must carry only its exact operation handle")


def reverify_admission_jit_interfaces(
    context: AdmissionVerificationContext,
    *,
    provider: JITCodeProvider,
    operation: ModelOperation | None,
) -> tuple[VerifiedRuntimeHandle, ...]:
    """Reload only one operation/component target and its exact prerequisite ports."""

    if not context.receipt_batch.interface_receipts:
        raise ValueError("biological execution requires at least one admitted interface receipt")
    matching_targets = tuple(
        target
        for target in context.query_prerequisite_report.targets
        if target.operation is operation
    )
    if len(matching_targets) != 1:
        raise ValueError("JIT execution requires one exact query-prerequisite target")
    target = matching_targets[0]
    expected_keys = {
        (ImplementationReceiptTargetKind.PORT, prerequisite.port.value)
        for prerequisite in target.required_ports
    }
    if operation is not None:
        expected_keys.add((ImplementationReceiptTargetKind.RUNTIME_OPERATION, operation.value))
    receipt_by_key: dict[tuple[ImplementationReceiptTargetKind, str], LoadedInterfaceReceipt] = {}
    for receipt in context.receipt_batch.interface_receipts:
        key = (receipt.requirement.target_kind, receipt.requirement.target_id)
        if key in receipt_by_key:
            raise ValueError("JIT interface receipts repeat one implementation target")
        receipt_by_key[key] = receipt
    missing = expected_keys - set(receipt_by_key)
    if missing:
        raise ValueError("JIT interface receipts omit an authorized prerequisite target")
    selected_receipts = tuple(
        receipt_by_key[key]
        for key in sorted(expected_keys, key=lambda item: (item[0].value, item[1]))
    )
    handles: list[VerifiedRuntimeHandle] = []
    for receipt in selected_receipts:
        resolved_code_content = provider(receipt)
        interface_name = receipt.requirement.implementation.interface
        trusted_interface = context.runtime_interfaces.get(interface_name)
        if trusted_interface is None:
            raise ValueError("JIT execution lacks the exact trusted runtime interface")
        loader_fingerprint = receipt.isolated_loader_observation.loader_fingerprint
        trusted_loader = context.jit_loaders.get(loader_fingerprint)
        if trusted_loader is None:
            raise ValueError("JIT execution lacks the exact trusted isolated loader")
        handles.append(
            reverify_jit_loaded_interface(
                receipt,
                resolved_code_content=resolved_code_content,
                trusted_runtime_interface=trusted_interface,
                trusted_jit_loader=trusted_loader,
            )
        )
    receipt_fingerprints = tuple(receipt.receipt_fingerprint for receipt in selected_receipts)
    handle_fingerprints = tuple(handle.receipt_fingerprint for handle in handles)
    if handle_fingerprints != receipt_fingerprints:
        raise ValueError("JIT runtime handles do not exactly cover the admitted interfaces")
    return tuple(handles)


def require_exact_admission_receipts(
    context: AdmissionVerificationContext,
    *,
    bundle: BiologicalModelBundleContract,
    benchmark: BenchmarkArtifact,
    training_run: TrainingRunBinding | None,
    validation_evidence: Sequence[ValidationEvidenceBinding],
) -> AdmissionReceiptBatchReport:
    """Rebind byte and loaded-interface receipts to every exact execution artifact."""

    additional = admission_execution_artifacts(
        benchmark=benchmark,
        training_run=training_run,
        validation_evidence=validation_evidence,
    )
    report = require_exact_receipt_batch_coverage(
        bundle,
        context.receipt_batch,
        trusted_verifiers=context.trusted_verifiers,
        runtime_interfaces=context.runtime_interfaces,
        additional_required_artifacts=additional,
    )
    authoritative_receipts = {
        receipt.artifact.fingerprint: receipt.receipt_fingerprint
        for receipt in report.artifact_receipts
    }
    if context.validation_result_batch is not None:
        for semantic_receipt in context.validation_result_batch.receipts:
            for artifact_receipt in semantic_receipt.artifact_receipts:
                expected = authoritative_receipts.get(artifact_receipt.artifact.fingerprint)
                if expected != artifact_receipt.receipt_fingerprint:
                    raise ValueError(
                        "validation semantics must reuse the authoritative exact-byte receipt"
                    )
    return report


def require_exact_query_prerequisites(
    context: AdmissionVerificationContext,
    *,
    query: StateQuery,
    support_envelope: BiologicalSupportEnvelope,
    bundle: BiologicalModelBundleContract,
) -> QueryDerivedPrerequisiteReport:
    """Recompile conditional prerequisites and require a structurally satisfied result."""

    report = verify_query_prerequisite_report(
        context.query_prerequisite_report,
        query=query,
        support_envelope=support_envelope,
        bundle=bundle,
    )
    if not report.structurally_satisfied:
        raise ValueError("query-derived biological prerequisites are not structurally satisfied")
    return report


def require_exact_admission_validation_results(
    context: AdmissionVerificationContext,
    *,
    bundle: BiologicalModelBundleContract,
    support_envelope: BiologicalSupportEnvelope,
    validation_evidence: Sequence[ValidationEvidenceBinding],
) -> ValidationResultReceiptBatch:
    """Rebind typed semantic results; failed results remain verified but do not pass."""

    if context.validation_result_batch is None:
        raise ValueError("admission context has no typed validation-result batch")
    return require_exact_validation_result_coverage(
        bundle,
        support_envelope,
        validation_evidence,
        context.validation_result_batch,
        trusted_verifiers=context.trusted_verifiers,
        authoritative_receipt_batch=context.receipt_batch,
    )


__all__ = [
    "REQUIRED_VALIDATION_CRITERIA",
    "VALIDATION_RESULT_MEDIA_TYPE",
    "VALIDATION_RESULT_SCHEMA_VERSION",
    "AdmissionVerificationContext",
    "BiologicalExecutionAuthorization",
    "JITCodeProvider",
    "ValidationCriterionResult",
    "ValidationResultManifest",
    "ValidationResultReceiptBatch",
    "ValidationResultVerificationReceipt",
    "ValidationSemanticCriterion",
    "admission_execution_artifacts",
    "build_validation_result_receipt_batch",
    "issue_validation_result_verification_receipt",
    "require_exact_admission_receipts",
    "require_exact_admission_validation_results",
    "require_exact_query_prerequisites",
    "require_exact_validation_result_coverage",
    "required_validation_criteria",
    "reverify_admission_jit_interfaces",
]
