"""Trusted, content-addressed receipts for future biological runtime admission.

Bundle declarations name artifacts and Python entry points, but names are not execution evidence.
This module defines immutable receipts for two narrower observations:

* a verifier read bytes whose digest and length exactly match a declared artifact; and
* a verifier loaded the exact declared Python entry point from the exact declared code artifact
  and observed the complete required interface signature set.

There are intentionally no caller-supplied ``verified`` or ``complete`` booleans.  Receipt and
batch fingerprints are recomputed from canonical payloads, and a batch is constructible only when
it covers every exact requirement once.  Trust in the named verifier and its evidence artifact is
an external policy decision; these contracts make that decision auditable rather than implicit.
"""

from __future__ import annotations

import hmac
import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import BinaryIO, Literal, TypeAlias, cast

from pydantic import ConfigDict, Field, field_validator, model_validator

from cellstate.data.benchmarks import ContentAddressedArtifact
from cellstate.data.manifests import SourceArtifact
from cellstate.domain.common import SchemaModel, canonical_fingerprint, canonical_json_bytes

from .contracts import (
    BiologicalModelBundleContract,
    BiologicalStagePort,
    ModelOperation,
    PortDisposition,
    PortImplementationBinding,
    PortImplementationKind,
)

AdmissionReceiptSchemaVersion = Literal["0.1-experimental"]
ADMISSION_RECEIPT_SCHEMA_VERSION: AdmissionReceiptSchemaVersion = "0.1-experimental"
_SHA256_PATTERN = r"^[a-fA-F0-9]{64}$"


class AdmissionReceiptModel(SchemaModel):
    """Strict base for experimental admission receipts."""

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


def _canonical_sha256(value: str) -> str:
    return value.casefold()


def _require_sorted_unique_text(
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


def _require_utc(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")
    return value.astimezone(UTC)


class AdmissionArtifactKind(StrEnum):
    """Namespace of an exact byte declaration consumed during admission."""

    CONTENT_ADDRESSED_ARTIFACT = "content_addressed_artifact"
    DATASET_SOURCE_ARTIFACT = "dataset_source_artifact"


class AdmissionArtifactReference(AdmissionReceiptModel):
    """Normalized exact reference to either a contract artifact or real-data source."""

    artifact_kind: AdmissionArtifactKind
    content_addressed_artifact: ContentAddressedArtifact | None = None
    dataset_source_artifact: SourceArtifact | None = None

    @model_validator(mode="after")
    def exactly_one_typed_artifact_is_present(self) -> AdmissionArtifactReference:
        if self.artifact_kind is AdmissionArtifactKind.CONTENT_ADDRESSED_ARTIFACT:
            if self.content_addressed_artifact is None or self.dataset_source_artifact is not None:
                raise ValueError("content-addressed references require only their typed artifact")
        elif self.dataset_source_artifact is None or self.content_addressed_artifact is not None:
            raise ValueError("dataset-source references require only their typed source artifact")
        return self

    @property
    def reference_id(self) -> str:
        if self.content_addressed_artifact is not None:
            return self.content_addressed_artifact.artifact_id
        assert self.dataset_source_artifact is not None
        return self.dataset_source_artifact.source_id

    @property
    def target_key(self) -> str:
        return f"{self.artifact_kind.value}:{self.reference_id}"

    @property
    def uri(self) -> str:
        if self.content_addressed_artifact is not None:
            return self.content_addressed_artifact.uri
        assert self.dataset_source_artifact is not None
        return self.dataset_source_artifact.uri

    @property
    def sha256(self) -> str:
        if self.content_addressed_artifact is not None:
            return self.content_addressed_artifact.sha256
        assert self.dataset_source_artifact is not None
        return self.dataset_source_artifact.sha256

    @property
    def byte_count(self) -> int:
        if self.content_addressed_artifact is not None:
            return self.content_addressed_artifact.byte_count
        assert self.dataset_source_artifact is not None
        return self.dataset_source_artifact.byte_count

    @property
    def media_type(self) -> str:
        if self.content_addressed_artifact is not None:
            return self.content_addressed_artifact.media_type
        assert self.dataset_source_artifact is not None
        return self.dataset_source_artifact.media_type

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


ArtifactDeclaration = ContentAddressedArtifact | SourceArtifact
ObservedByteSource: TypeAlias = bytes | BinaryIO | Iterable[bytes]


def admission_artifact_reference(artifact: ArtifactDeclaration) -> AdmissionArtifactReference:
    """Normalize an exact declaration without dropping dataset accession/release metadata."""

    if isinstance(artifact, ContentAddressedArtifact):
        return AdmissionArtifactReference(
            artifact_kind=AdmissionArtifactKind.CONTENT_ADDRESSED_ARTIFACT,
            content_addressed_artifact=artifact,
        )
    return AdmissionArtifactReference(
        artifact_kind=AdmissionArtifactKind.DATASET_SOURCE_ARTIFACT,
        dataset_source_artifact=artifact,
    )


def _artifact_identity(artifact: AdmissionArtifactReference) -> str:
    return artifact.fingerprint


def _require_canonical_artifacts(
    artifacts: tuple[AdmissionArtifactReference, ...],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[AdmissionArtifactReference, ...]:
    if not allow_empty and not artifacts:
        raise ValueError(f"{name} must not be empty")
    target_keys = tuple(artifact.target_key for artifact in artifacts)
    _require_sorted_unique_text(target_keys, name=f"{name} target keys", allow_empty=allow_empty)
    identities = tuple(_artifact_identity(artifact) for artifact in artifacts)
    if len(identities) != len(set(identities)):
        raise ValueError(f"{name} must not repeat an exact artifact reference")
    return artifacts


def _require_canonical_evidence_artifacts(
    artifacts: tuple[ContentAddressedArtifact, ...],
    *,
    name: str,
) -> tuple[ContentAddressedArtifact, ...]:
    if not artifacts:
        raise ValueError(f"{name} must not be empty")
    artifact_ids = tuple(artifact.artifact_id for artifact in artifacts)
    _require_sorted_unique_text(artifact_ids, name=f"{name} IDs", allow_empty=False)
    identities = tuple(
        canonical_fingerprint(artifact.model_dump(mode="json")) for artifact in artifacts
    )
    if len(identities) != len(set(identities)):
        raise ValueError(f"{name} must not repeat an exact artifact reference")
    return artifacts


class AdmissionVerifierCapability(StrEnum):
    """Exact observation class a verifier implementation is trusted to attest."""

    ARTIFACT_BYTE_RESOLUTION = "artifact_byte_resolution"
    EXECUTION_SOURCE_SELECTION = "execution_source_selection"
    LOADED_INTERFACE_VERIFICATION = "loaded_interface_verification"
    VALIDATION_RESULT_SEMANTICS = "validation_result_semantics"


class AdmissionVerifierIdentity(AdmissionReceiptModel):
    """Exact versioned identity of the code that issued an admission receipt."""

    verifier_id: str = Field(min_length=1)
    verifier_version: str = Field(min_length=1)
    code_artifact: ContentAddressedArtifact
    entrypoint: str = Field(min_length=1)
    runtime: str = Field(min_length=1)
    capabilities: tuple[AdmissionVerifierCapability, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identity_is_canonical(self) -> AdmissionVerifierIdentity:
        _canonical_text(self.verifier_id, name="verifier ID")
        _canonical_text(self.verifier_version, name="verifier version")
        _canonical_text(self.entrypoint, name="verifier entrypoint")
        _canonical_text(self.runtime, name="verifier runtime")
        capability_values = tuple(capability.value for capability in self.capabilities)
        _require_sorted_unique_text(
            capability_values,
            name="verifier capabilities",
            allow_empty=False,
        )
        return self

    @property
    def fingerprint(self) -> str:
        """Stable identity including the verifier's exact code artifact."""

        return canonical_fingerprint(self.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class TrustedAdmissionVerifier:
    """Non-serializable trust root used to authenticate receipt issuance.

    The secret is deliberately excluded from every Pydantic contract, fingerprint, representation,
    and JSON payload.  Deployments are responsible for provisioning it to an isolated verifier.
    """

    identity: AdmissionVerifierIdentity
    key_id: str
    secret: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _canonical_text(self.key_id, name="verifier attestation key ID")
        if not isinstance(self.secret, bytes) or len(self.secret) < 32:
            raise ValueError("verifier attestation secrets must contain at least 32 bytes")


@dataclass(frozen=True, slots=True)
class TrustedRuntimeInterface:
    """Application-owned interface registry entry, never supplied by a report submitter."""

    declared_interface: str
    interface_artifact: ContentAddressedArtifact
    runtime_interface: type[object] = field(repr=False)

    def __post_init__(self) -> None:
        _canonical_text(self.declared_interface, name="trusted runtime interface")
        if not inspect.isclass(self.runtime_interface):
            raise ValueError("trusted runtime interface registry entries must be classes")

    @property
    def fingerprint(self) -> str:
        module, qualname = _runtime_interface_identity(self.runtime_interface)
        members = _runtime_interface_members(self.runtime_interface)
        return canonical_fingerprint(
            {
                "declared_interface": self.declared_interface,
                "interface_artifact": self.interface_artifact.model_dump(mode="json"),
                "runtime_interface_module": module,
                "runtime_interface_qualname": qualname,
                "required_member_signatures": members,
            }
        )


class ReceiptAttestation(AdmissionReceiptModel):
    """Keyed authentication of a receipt payload by an external verifier trust root."""

    algorithm: Literal["hmac-sha256-v1"] = "hmac-sha256-v1"
    key_id: str = Field(min_length=1)
    attested_payload_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    authentication_tag: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("key_id")
    @classmethod
    def key_id_is_canonical(cls, value: str) -> str:
        return _canonical_text(value, name="verifier attestation key ID")

    @field_validator("attested_payload_fingerprint", "authentication_tag")
    @classmethod
    def hashes_are_canonical(cls, value: str) -> str:
        return _canonical_sha256(value)


def attest_canonical_payload(
    payload: Mapping[str, object],
    *,
    trusted_verifier: TrustedAdmissionVerifier,
    required_capability: AdmissionVerifierCapability,
) -> ReceiptAttestation:
    """Authenticate a bounded canonical payload with a capability-scoped trust root."""

    if required_capability not in trusted_verifier.identity.capabilities:
        raise ValueError(
            f"verifier lacks required {required_capability.value!r} attestation capability"
        )
    payload_fingerprint = canonical_fingerprint(payload)
    authentication_tag = hmac.new(
        trusted_verifier.secret,
        canonical_json_bytes(payload),
        sha256,
    ).hexdigest()
    return ReceiptAttestation(
        key_id=trusted_verifier.key_id,
        attested_payload_fingerprint=payload_fingerprint,
        authentication_tag=authentication_tag,
    )


def require_valid_canonical_attestation(
    payload: Mapping[str, object],
    attestation: ReceiptAttestation,
    *,
    verifier_identity: AdmissionVerifierIdentity,
    trusted_verifier: TrustedAdmissionVerifier,
    required_capability: AdmissionVerifierCapability,
) -> None:
    """Validate identity, capability, payload hash, and keyed authentication exactly."""

    identity = AdmissionVerifierIdentity.model_validate(verifier_identity.model_dump(mode="python"))
    if identity != trusted_verifier.identity:
        raise ValueError("attestation verifier is absent from the exact external trust root")
    if required_capability not in identity.capabilities:
        raise ValueError(
            f"verifier lacks required {required_capability.value!r} attestation capability"
        )
    if attestation.key_id != trusted_verifier.key_id:
        raise ValueError("attestation key is absent from the exact external trust root")
    expected_payload_fingerprint = canonical_fingerprint(payload)
    if not hmac.compare_digest(
        attestation.attested_payload_fingerprint,
        expected_payload_fingerprint,
    ):
        raise ValueError("attestation payload fingerprint does not match")
    expected_tag = hmac.new(
        trusted_verifier.secret,
        canonical_json_bytes(payload),
        sha256,
    ).hexdigest()
    if not hmac.compare_digest(attestation.authentication_tag, expected_tag):
        raise ValueError("attestation authentication failed")


def _observe_byte_source(source: ObservedByteSource) -> tuple[str, int]:
    """Incrementally hash a byte source without materializing large artifacts in memory."""

    digest = sha256()
    observed_byte_count = 0

    def consume(chunk: bytes) -> None:
        nonlocal observed_byte_count
        if not isinstance(chunk, bytes):
            raise TypeError("observed artifact streams must yield bytes")
        digest.update(chunk)
        observed_byte_count += len(chunk)

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
    if observed_byte_count == 0:
        raise ValueError("observed artifact byte source must not be empty")
    return digest.hexdigest(), observed_byte_count


def _seal_byte_source(source: ObservedByteSource, *, expected_byte_count: int) -> bytes:
    """Materialize one immutable snapshot for a trusted JIT loader and its hash check.

    Code loading cannot safely inspect one object while hashing an unrelated stream.  This helper
    consumes the resolver output once, rejects overlong streams early, and returns the exact bytes
    that both the verifier and the application-owned loader must use.
    """

    sealed = bytearray()

    def consume(chunk: bytes) -> None:
        if not isinstance(chunk, bytes):
            raise TypeError("JIT code streams must yield bytes")
        sealed.extend(chunk)
        if len(sealed) > expected_byte_count:
            raise ValueError("JIT code bytes exceed the admitted implementation byte count")

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
    if not sealed:
        raise ValueError("JIT code byte source must not be empty")
    if len(sealed) != expected_byte_count:
        raise ValueError("JIT code bytes do not match the admitted implementation byte count")
    return bytes(sealed)


class ExecutionSourceSelectionReceipt(AdmissionReceiptModel):
    """Authenticated legal/scientific selection of exact real-data execution bytes.

    The selection is external to a bundle because a bundle contains only the content-addressed
    benchmark declaration.  An application-owned workflow resolver must derive ``sources`` from
    typed dataset-assessment resolutions.  Its exact resolution artifacts are included in byte
    coverage; arbitrary manifest review sources are never inferred as execution inputs.
    """

    schema_version: AdmissionReceiptSchemaVersion = ADMISSION_RECEIPT_SCHEMA_VERSION
    receipt_kind: Literal["execution_source_selection"] = "execution_source_selection"
    selection_id: str = Field(min_length=1)
    bundle_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    workflow_resolution_artifacts: tuple[ContentAddressedArtifact, ...] = Field(min_length=1)
    sources: tuple[SourceArtifact, ...] = Field(min_length=1)
    selector: AdmissionVerifierIdentity
    selector_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    issued_at: datetime
    attestation: ReceiptAttestation
    selection_fingerprint: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("selection_id")
    @classmethod
    def selection_id_is_canonical(cls, value: str) -> str:
        return _canonical_text(value, name="execution-source selection ID")

    @field_validator(
        "bundle_fingerprint",
        "selector_fingerprint",
        "selection_fingerprint",
    )
    @classmethod
    def fingerprints_are_canonical(cls, value: str) -> str:
        return _canonical_sha256(value)

    @field_validator("issued_at")
    @classmethod
    def issued_at_is_utc(cls, value: datetime) -> datetime:
        return _require_utc(value, name="execution-source selection issuance time")

    @field_validator("workflow_resolution_artifacts")
    @classmethod
    def workflow_artifacts_are_canonical(
        cls,
        value: tuple[ContentAddressedArtifact, ...],
    ) -> tuple[ContentAddressedArtifact, ...]:
        return _require_canonical_evidence_artifacts(
            value,
            name="execution-source workflow resolution artifacts",
        )

    @field_validator("sources")
    @classmethod
    def sources_are_canonical(
        cls,
        value: tuple[SourceArtifact, ...],
    ) -> tuple[SourceArtifact, ...]:
        source_ids = tuple(source.source_id for source in value)
        _require_sorted_unique_text(
            source_ids,
            name="execution-source IDs",
            allow_empty=False,
        )
        identities = tuple(
            canonical_fingerprint(source.model_dump(mode="json")) for source in value
        )
        if len(identities) != len(set(identities)):
            raise ValueError("execution sources must not repeat an exact source artifact")
        return value

    @model_validator(mode="after")
    def selection_is_exact(self) -> ExecutionSourceSelectionReceipt:
        if self.selector_fingerprint != self.selector.fingerprint:
            raise ValueError("execution-source selector fingerprint does not match its identity")
        if AdmissionVerifierCapability.EXECUTION_SOURCE_SELECTION not in self.selector.capabilities:
            raise ValueError("selector lacks execution-source selection capability")
        attested_payload = self.model_dump(
            mode="python",
            exclude={"attestation", "selection_fingerprint"},
        )
        if self.attestation.attested_payload_fingerprint != canonical_fingerprint(attested_payload):
            raise ValueError("execution-source attestation binds a different selection")
        expected = canonical_fingerprint(
            self.model_dump(mode="python", exclude={"selection_fingerprint"})
        )
        if self.selection_fingerprint != expected:
            raise ValueError("execution-source selection fingerprint does not match its payload")
        return self


def issue_execution_source_selection_receipt(
    *,
    selection_id: str,
    bundle: BiologicalModelBundleContract,
    workflow_resolution_artifacts: Sequence[ContentAddressedArtifact],
    sources: Sequence[SourceArtifact],
    trusted_selector: TrustedAdmissionVerifier,
    issued_at: datetime,
) -> ExecutionSourceSelectionReceipt:
    """Issue an authenticated output from the trusted assessment/permission workflow.

    The trusted selector, not the report submitter, is responsible for deriving ``sources`` from
    typed ``DatasetAssessmentResolution.data_source_ids`` and matching exact manifests.  The
    resolver's canonical resolution artifacts are mandatory so the derivation itself receives
    exact-byte coverage in the admission batch.
    """

    bundle = BiologicalModelBundleContract.model_validate(bundle.model_dump(mode="python"))
    source_tuple = tuple(sorted(sources, key=lambda source: source.source_id))
    workflow_tuple = tuple(
        sorted(workflow_resolution_artifacts, key=lambda artifact: artifact.artifact_id)
    )
    payload = {
        "schema_version": ADMISSION_RECEIPT_SCHEMA_VERSION,
        "receipt_kind": "execution_source_selection",
        "selection_id": selection_id,
        "bundle_fingerprint": bundle.fingerprint,
        "workflow_resolution_artifacts": workflow_tuple,
        "sources": source_tuple,
        "selector": trusted_selector.identity,
        "selector_fingerprint": trusted_selector.identity.fingerprint,
        "issued_at": issued_at,
    }
    attestation = attest_canonical_payload(
        payload,
        trusted_verifier=trusted_selector,
        required_capability=AdmissionVerifierCapability.EXECUTION_SOURCE_SELECTION,
    )
    selection_payload = {**payload, "attestation": attestation}
    return ExecutionSourceSelectionReceipt.model_validate(
        {
            **selection_payload,
            "selection_fingerprint": canonical_fingerprint(selection_payload),
        }
    )


def require_valid_execution_source_selection(
    selection: ExecutionSourceSelectionReceipt,
    *,
    trusted_selector: TrustedAdmissionVerifier,
) -> ExecutionSourceSelectionReceipt:
    """Authenticate a source selection against its external workflow-selector trust root."""

    selection = ExecutionSourceSelectionReceipt.model_validate(selection.model_dump(mode="python"))
    attested_payload = selection.model_dump(
        mode="python",
        exclude={"attestation", "selection_fingerprint"},
    )
    require_valid_canonical_attestation(
        attested_payload,
        selection.attestation,
        verifier_identity=selection.selector,
        trusted_verifier=trusted_selector,
        required_capability=AdmissionVerifierCapability.EXECUTION_SOURCE_SELECTION,
    )
    return selection


class ArtifactByteObservation(AdmissionReceiptModel):
    """Digest and length computed from one concrete byte stream."""

    source_uri: str = Field(min_length=1)
    observed_sha256: str = Field(pattern=_SHA256_PATTERN)
    observed_byte_count: int = Field(gt=0)

    @field_validator("source_uri")
    @classmethod
    def source_uri_is_canonical(cls, value: str) -> str:
        return _canonical_text(value, name="observed artifact source URI")

    @field_validator("observed_sha256")
    @classmethod
    def sha256_is_canonical(cls, value: str) -> str:
        return _canonical_sha256(value)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


class ArtifactResolutionReceipt(AdmissionReceiptModel):
    """Positive receipt proving that observed bytes exactly match one declaration."""

    schema_version: AdmissionReceiptSchemaVersion = ADMISSION_RECEIPT_SCHEMA_VERSION
    receipt_kind: Literal["artifact_resolution"] = "artifact_resolution"
    receipt_id: str = Field(min_length=1)
    artifact: AdmissionArtifactReference
    observation: ArtifactByteObservation
    verifier: AdmissionVerifierIdentity
    verifier_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    issued_at: datetime
    evidence_artifacts: tuple[ContentAddressedArtifact, ...] = Field(
        min_length=1,
        description=(
            "Non-authoritative audit-log references; they never satisfy byte or interface "
            "coverage and must not be treated as readiness evidence."
        ),
    )
    attestation: ReceiptAttestation
    receipt_fingerprint: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("receipt_id")
    @classmethod
    def receipt_id_is_canonical(cls, value: str) -> str:
        return _canonical_text(value, name="artifact-resolution receipt ID")

    @field_validator("verifier_fingerprint", "receipt_fingerprint")
    @classmethod
    def fingerprints_are_canonical(cls, value: str) -> str:
        return _canonical_sha256(value)

    @field_validator("issued_at")
    @classmethod
    def issued_at_is_utc(cls, value: datetime) -> datetime:
        return _require_utc(value, name="artifact-resolution issuance time")

    @field_validator("evidence_artifacts")
    @classmethod
    def evidence_is_canonical(
        cls, value: tuple[ContentAddressedArtifact, ...]
    ) -> tuple[ContentAddressedArtifact, ...]:
        return _require_canonical_evidence_artifacts(value, name="artifact-resolution evidence")

    @model_validator(mode="after")
    def receipt_is_exact(self) -> ArtifactResolutionReceipt:
        if self.observation.source_uri != self.artifact.uri:
            raise ValueError("observed byte source URI must exactly match the artifact declaration")
        if self.observation.observed_sha256 != self.artifact.sha256:
            raise ValueError("observed byte SHA-256 must exactly match the artifact declaration")
        if self.observation.observed_byte_count != self.artifact.byte_count:
            raise ValueError("observed byte count must exactly match the artifact declaration")
        if self.verifier_fingerprint != self.verifier.fingerprint:
            raise ValueError("artifact-resolution verifier fingerprint does not match its identity")
        if AdmissionVerifierCapability.ARTIFACT_BYTE_RESOLUTION not in self.verifier.capabilities:
            raise ValueError("receipt verifier lacks artifact-byte resolution capability")
        if self.artifact.reference_id in {
            evidence.artifact_id for evidence in self.evidence_artifacts
        }:
            raise ValueError("resolution evidence must be distinct from the resolved artifact")
        attested_payload = self.model_dump(
            mode="python",
            exclude={"attestation", "receipt_fingerprint"},
        )
        if self.attestation.attested_payload_fingerprint != canonical_fingerprint(attested_payload):
            raise ValueError("artifact-resolution attestation binds a different payload")
        expected = canonical_fingerprint(
            self.model_dump(mode="python", exclude={"receipt_fingerprint"})
        )
        if self.receipt_fingerprint != expected:
            raise ValueError("artifact-resolution receipt fingerprint does not match its payload")
        return self


class ImplementationReceiptTargetKind(StrEnum):
    """Kind of executable declaration covered by one interface receipt."""

    PORT = "port"
    RUNTIME_OPERATION = "runtime_operation"


class ImplementationReceiptRequirement(AdmissionReceiptModel):
    """Exact implementation target whose loaded interface must be verified."""

    target_kind: ImplementationReceiptTargetKind
    target_id: str = Field(min_length=1)
    bundle_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    implementation_scope_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    implementation: PortImplementationBinding
    interface_artifact: ContentAddressedArtifact
    trusted_interface_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    runtime_interface_module: str = Field(min_length=1)
    runtime_interface_qualname: str = Field(min_length=1)
    required_member_signatures: tuple[str, ...] = Field(min_length=1)
    interface_contract_fingerprint: str = Field(pattern=_SHA256_PATTERN)

    @field_validator(
        "target_id",
        "runtime_interface_module",
        "runtime_interface_qualname",
    )
    @classmethod
    def target_id_is_canonical(cls, value: str) -> str:
        return _canonical_text(value, name="implementation target ID")

    @field_validator(
        "bundle_fingerprint",
        "implementation_scope_fingerprint",
        "interface_contract_fingerprint",
        "trusted_interface_fingerprint",
    )
    @classmethod
    def fingerprints_are_canonical(cls, value: str) -> str:
        return _canonical_sha256(value)

    @field_validator("required_member_signatures")
    @classmethod
    def member_signatures_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _require_sorted_unique_text(
            value,
            name="required interface member signatures",
            allow_empty=False,
        )

    @model_validator(mode="after")
    def target_is_executable_and_typed(self) -> ImplementationReceiptRequirement:
        valid_target_ids = (
            {port.value for port in BiologicalStagePort}
            if self.target_kind is ImplementationReceiptTargetKind.PORT
            else {operation.value for operation in ModelOperation}
        )
        if self.target_id not in valid_target_ids:
            raise ValueError("implementation target ID does not belong to its declared target kind")
        if self.implementation.kind is not PortImplementationKind.PYTHON_ENTRY_POINT:
            raise ValueError("interface receipts cannot cover specification-only implementations")
        if self.implementation.entrypoint is None:
            raise ValueError("interface receipt requirements need an exact Python entry point")
        expected_interface_fingerprint = _interface_contract_fingerprint(
            self.implementation.interface,
            self.runtime_interface_module,
            self.runtime_interface_qualname,
            self.required_member_signatures,
        )
        if self.interface_contract_fingerprint != expected_interface_fingerprint:
            raise ValueError("interface contract fingerprint does not match its exact declaration")
        expected_trusted_interface = canonical_fingerprint(
            {
                "declared_interface": self.implementation.interface,
                "interface_artifact": self.interface_artifact.model_dump(mode="json"),
                "runtime_interface_module": self.runtime_interface_module,
                "runtime_interface_qualname": self.runtime_interface_qualname,
                "required_member_signatures": self.required_member_signatures,
            }
        )
        if self.trusted_interface_fingerprint != expected_trusted_interface:
            raise ValueError(
                "trusted interface fingerprint does not match its exact registry entry"
            )
        return self

    @property
    def target_key(self) -> str:
        return f"{self.target_kind.value}:{self.target_id}"

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


class LoadedObjectKind(StrEnum):
    """Serializable runtime kind observed after resolving a Python entry point."""

    CLASS = "class"
    FUNCTION = "function"
    INSTANCE = "instance"


class LoadedObjectIdentity(AdmissionReceiptModel):
    """Exact serializable identity of the object loaded from declared code bytes."""

    entrypoint: str = Field(min_length=1)
    module: str = Field(min_length=1)
    qualname: str = Field(min_length=1)
    object_kind: LoadedObjectKind
    loaded_code_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("entrypoint", "module", "qualname")
    @classmethod
    def names_are_canonical(cls, value: str) -> str:
        return _canonical_text(value, name="loaded-object identity")

    @field_validator("loaded_code_sha256")
    @classmethod
    def sha256_is_canonical(cls, value: str) -> str:
        return _canonical_sha256(value)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))


class InterfaceVerificationMethod(StrEnum):
    """Auditable mechanism used to compare an object with its declared interface."""

    ABSTRACT_BASE_CLASS = "abstract_base_class"
    INSPECTED_SIGNATURE_SET = "inspected_signature_set"
    RUNTIME_CHECKABLE_PROTOCOL = "runtime_checkable_protocol"


def _interface_contract_fingerprint(
    interface: str,
    runtime_interface_module: str,
    runtime_interface_qualname: str,
    members: tuple[str, ...],
) -> str:
    return canonical_fingerprint(
        {
            "interface": interface,
            "runtime_interface_module": runtime_interface_module,
            "runtime_interface_qualname": runtime_interface_qualname,
            "member_signatures": members,
        }
    )


class InterfaceConformanceObservation(AdmissionReceiptModel):
    """Exact positive structural observation emitted by a named interface verifier."""

    verification_method: InterfaceVerificationMethod
    declared_interface: str = Field(min_length=1)
    runtime_interface_module: str = Field(min_length=1)
    runtime_interface_qualname: str = Field(min_length=1)
    loaded_object_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    required_member_signatures: tuple[str, ...] = Field(min_length=1)
    observed_matching_member_signatures: tuple[str, ...] = Field(min_length=1)
    interface_contract_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    observed_contract_fingerprint: str = Field(pattern=_SHA256_PATTERN)

    @field_validator(
        "declared_interface",
        "runtime_interface_module",
        "runtime_interface_qualname",
    )
    @classmethod
    def interface_is_canonical(cls, value: str) -> str:
        return _canonical_text(value, name="declared interface")

    @field_validator(
        "loaded_object_fingerprint",
        "interface_contract_fingerprint",
        "observed_contract_fingerprint",
    )
    @classmethod
    def fingerprints_are_canonical(cls, value: str) -> str:
        return _canonical_sha256(value)

    @field_validator("required_member_signatures", "observed_matching_member_signatures")
    @classmethod
    def member_signatures_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _require_sorted_unique_text(
            value,
            name="interface member signatures",
            allow_empty=False,
        )

    @model_validator(mode="after")
    def observed_contract_exactly_matches_requirement(self) -> InterfaceConformanceObservation:
        if self.observed_matching_member_signatures != self.required_member_signatures:
            raise ValueError(
                "observed interface members must exactly match the required signatures"
            )
        expected = _interface_contract_fingerprint(
            self.declared_interface,
            self.runtime_interface_module,
            self.runtime_interface_qualname,
            self.required_member_signatures,
        )
        observed = _interface_contract_fingerprint(
            self.declared_interface,
            self.runtime_interface_module,
            self.runtime_interface_qualname,
            self.observed_matching_member_signatures,
        )
        if self.interface_contract_fingerprint != expected:
            raise ValueError("declared interface fingerprint does not match its signature set")
        if self.observed_contract_fingerprint != observed:
            raise ValueError("observed interface fingerprint does not match its signature set")
        if self.observed_contract_fingerprint != self.interface_contract_fingerprint:
            raise ValueError("loaded object does not conform to the exact declared interface")
        return self


class LoadedInterfaceReceipt(AdmissionReceiptModel):
    """Positive receipt binding a loaded object to one exact implementation requirement."""

    schema_version: AdmissionReceiptSchemaVersion = ADMISSION_RECEIPT_SCHEMA_VERSION
    receipt_kind: Literal["loaded_interface"] = "loaded_interface"
    receipt_id: str = Field(min_length=1)
    requirement: ImplementationReceiptRequirement
    loaded_object: LoadedObjectIdentity
    conformance: InterfaceConformanceObservation
    isolated_loader_observation: IsolatedLoadedInterfaceObservation
    execution_recheck_requirement: Literal["reload_exact_bytes_and_reverify_before_execution"] = (
        "reload_exact_bytes_and_reverify_before_execution"
    )
    verifier: AdmissionVerifierIdentity
    verifier_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    issued_at: datetime
    evidence_artifacts: tuple[ContentAddressedArtifact, ...] = Field(
        min_length=1,
        description=(
            "Non-authoritative audit-log references; they never satisfy byte or interface "
            "coverage and must not be treated as readiness evidence."
        ),
    )
    attestation: ReceiptAttestation
    receipt_fingerprint: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("receipt_id")
    @classmethod
    def receipt_id_is_canonical(cls, value: str) -> str:
        return _canonical_text(value, name="loaded-interface receipt ID")

    @field_validator("verifier_fingerprint", "receipt_fingerprint")
    @classmethod
    def fingerprints_are_canonical(cls, value: str) -> str:
        return _canonical_sha256(value)

    @field_validator("issued_at")
    @classmethod
    def issued_at_is_utc(cls, value: datetime) -> datetime:
        return _require_utc(value, name="loaded-interface issuance time")

    @field_validator("evidence_artifacts")
    @classmethod
    def evidence_is_canonical(
        cls, value: tuple[ContentAddressedArtifact, ...]
    ) -> tuple[ContentAddressedArtifact, ...]:
        return _require_canonical_evidence_artifacts(value, name="loaded-interface evidence")

    @model_validator(mode="after")
    def receipt_is_exact(self) -> LoadedInterfaceReceipt:
        implementation = self.requirement.implementation
        if self.loaded_object.entrypoint != implementation.entrypoint:
            raise ValueError("loaded entry point must exactly match the implementation declaration")
        if self.loaded_object.loaded_code_sha256 != implementation.code_artifact.sha256:
            raise ValueError("loaded object code hash must exactly match its declared artifact")
        if self.conformance.declared_interface != implementation.interface:
            raise ValueError("verified interface must exactly match the implementation declaration")
        if (
            self.conformance.runtime_interface_module != self.requirement.runtime_interface_module
            or self.conformance.runtime_interface_qualname
            != self.requirement.runtime_interface_qualname
        ):
            raise ValueError("interface observation must bind the exact runtime interface object")
        if (
            self.conformance.required_member_signatures
            != self.requirement.required_member_signatures
            or self.conformance.interface_contract_fingerprint
            != self.requirement.interface_contract_fingerprint
        ):
            raise ValueError("interface observation must bind the exact required signature set")
        if self.conformance.loaded_object_fingerprint != self.loaded_object.fingerprint:
            raise ValueError("interface observation must bind the exact loaded object")
        if (
            self.isolated_loader_observation.loaded_object != self.loaded_object
            or self.isolated_loader_observation.conformance != self.conformance
            or self.isolated_loader_observation.requirement_fingerprint
            != self.requirement.fingerprint
        ):
            raise ValueError(
                "interface receipt must preserve the exact isolated-loader observation"
            )
        if self.verifier_fingerprint != self.verifier.fingerprint:
            raise ValueError("interface verifier fingerprint does not match its identity")
        if (
            AdmissionVerifierCapability.LOADED_INTERFACE_VERIFICATION
            not in self.verifier.capabilities
        ):
            raise ValueError("receipt verifier lacks loaded-interface verification capability")
        attested_payload = self.model_dump(
            mode="python",
            exclude={"attestation", "receipt_fingerprint"},
        )
        if self.attestation.attested_payload_fingerprint != canonical_fingerprint(attested_payload):
            raise ValueError("loaded-interface attestation binds a different payload")
        expected = canonical_fingerprint(
            self.model_dump(mode="python", exclude={"receipt_fingerprint"})
        )
        if self.receipt_fingerprint != expected:
            raise ValueError("loaded-interface receipt fingerprint does not match its payload")
        return self


def issue_artifact_resolution_receipt(
    *,
    receipt_id: str,
    artifact: ArtifactDeclaration,
    observed_content: ObservedByteSource,
    trusted_verifier: TrustedAdmissionVerifier,
    issued_at: datetime,
    evidence_artifacts: Sequence[ContentAddressedArtifact],
) -> ArtifactResolutionReceipt:
    """Hash concrete bytes and issue a receipt only when they match the declaration exactly."""

    reference = admission_artifact_reference(artifact)
    observed_sha256, observed_byte_count = _observe_byte_source(observed_content)
    observation = ArtifactByteObservation(
        source_uri=reference.uri,
        observed_sha256=observed_sha256,
        observed_byte_count=observed_byte_count,
    )
    evidence = tuple(evidence_artifacts)
    payload = {
        "schema_version": ADMISSION_RECEIPT_SCHEMA_VERSION,
        "receipt_kind": "artifact_resolution",
        "receipt_id": receipt_id,
        "artifact": reference,
        "observation": observation,
        "verifier": trusted_verifier.identity,
        "verifier_fingerprint": trusted_verifier.identity.fingerprint,
        "issued_at": issued_at,
        "evidence_artifacts": evidence,
    }
    attestation = attest_canonical_payload(
        payload,
        trusted_verifier=trusted_verifier,
        required_capability=AdmissionVerifierCapability.ARTIFACT_BYTE_RESOLUTION,
    )
    receipt_payload = {**payload, "attestation": attestation}
    return ArtifactResolutionReceipt.model_validate(
        {
            **receipt_payload,
            "receipt_fingerprint": canonical_fingerprint(receipt_payload),
        }
    )


def _call_shape(value: object) -> str:
    if not callable(value):
        raise ValueError("interface member is not callable")
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError) as error:
        raise ValueError("interface callable does not expose an inspectable signature") from error
    return canonical_fingerprint({"inspect_signature": str(signature)})


def _member_signature(owner: object, member_name: str) -> str:
    try:
        member = inspect.getattr_static(owner, member_name)
    except AttributeError as error:
        raise ValueError(f"loaded object is missing required member {member_name!r}") from error
    if isinstance(member, property):
        if member.fget is None:
            raise ValueError(f"interface property {member_name!r} has no readable getter")
        async_kind = "async" if inspect.iscoroutinefunction(member.fget) else "sync"
        return f"{member_name}|property|{async_kind}|{_call_shape(member.fget)}"
    if isinstance(member, classmethod):
        descriptor_kind = "class_method"
        member = member.__func__
    elif isinstance(member, staticmethod):
        descriptor_kind = "static_method"
        member = member.__func__
    elif inspect.isfunction(member):
        descriptor_kind = "instance_method"
    else:
        descriptor_kind = "callable_attribute"
    if callable(member):
        async_kind = "async" if inspect.iscoroutinefunction(member) else "sync"
        return f"{member_name}|{descriptor_kind}|{async_kind}|{_call_shape(member)}"
    return f"{member_name}|attribute"


def _runtime_interface_members(runtime_interface: type[object]) -> tuple[str, ...]:
    member_names: set[str] = set()
    for ancestor in runtime_interface.__mro__:
        if ancestor is object or ancestor.__module__ in {"abc", "typing"}:
            continue
        annotations = inspect.get_annotations(ancestor, eval_str=False)
        member_names.update(name for name in annotations if not name.startswith("_"))
        member_names.update(
            name
            for name, value in ancestor.__dict__.items()
            if not name.startswith("_")
            and (isinstance(value, (classmethod, property, staticmethod)) or callable(value))
        )
    if not member_names:
        raise ValueError("runtime interface must declare at least one public member")
    return tuple(
        sorted(_member_signature(runtime_interface, member_name) for member_name in member_names)
    )


def _runtime_interface_identity(runtime_interface: type[object]) -> tuple[str, str]:
    module = _canonical_text(runtime_interface.__module__, name="runtime interface module")
    qualname = _canonical_text(runtime_interface.__qualname__, name="runtime interface qualname")
    return module, qualname


def _loaded_object_identity(
    loaded_object: object,
    *,
    entrypoint: str,
    loaded_code_sha256: str,
) -> LoadedObjectIdentity:
    identity_owner: object
    if inspect.isclass(loaded_object):
        kind = LoadedObjectKind.CLASS
        identity_owner = loaded_object
    elif inspect.isfunction(loaded_object):
        kind = LoadedObjectKind.FUNCTION
        identity_owner = loaded_object
    else:
        kind = LoadedObjectKind.INSTANCE
        identity_owner = type(loaded_object)
    module = getattr(identity_owner, "__module__", None)
    qualname = getattr(identity_owner, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise ValueError("loaded object does not expose a canonical Python identity")
    return LoadedObjectIdentity(
        entrypoint=entrypoint,
        module=module,
        qualname=qualname,
        object_kind=kind,
        loaded_code_sha256=loaded_code_sha256,
    )


def _verification_method(
    runtime_interface: type[object],
    loaded_object: object,
) -> InterfaceVerificationMethod:
    if bool(getattr(runtime_interface, "_is_runtime_protocol", False)) and not inspect.isclass(
        loaded_object
    ):
        return InterfaceVerificationMethod.RUNTIME_CHECKABLE_PROTOCOL
    if inspect.isabstract(runtime_interface):
        return InterfaceVerificationMethod.ABSTRACT_BASE_CLASS
    return InterfaceVerificationMethod.INSPECTED_SIGNATURE_SET


def _require_nominal_conformance(
    loaded_object: object,
    runtime_interface: type[object],
    method: InterfaceVerificationMethod,
) -> None:
    if method is InterfaceVerificationMethod.INSPECTED_SIGNATURE_SET:
        return
    if inspect.isclass(loaded_object):
        try:
            conforms = issubclass(loaded_object, runtime_interface)
        except TypeError as error:
            raise ValueError(
                "loaded class cannot be checked against its runtime interface"
            ) from error
    else:
        try:
            conforms = isinstance(loaded_object, runtime_interface)
        except TypeError as error:
            raise ValueError("loaded object interface is not runtime-checkable") from error
    if not conforms:
        raise ValueError("loaded object fails nominal runtime interface conformance")


class IsolatedLoadedInterfaceObservation(AdmissionReceiptModel):
    """Authenticated bounded output from an application-owned isolated loader."""

    schema_version: AdmissionReceiptSchemaVersion = ADMISSION_RECEIPT_SCHEMA_VERSION
    observation_kind: Literal["isolated_loaded_interface"] = "isolated_loaded_interface"
    observation_id: str = Field(min_length=1)
    requirement_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    implementation_scope_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    implementation_code_artifact: ContentAddressedArtifact
    loaded_object: LoadedObjectIdentity
    conformance: InterfaceConformanceObservation
    loader: AdmissionVerifierIdentity
    loader_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    issued_at: datetime
    isolation_evidence_artifact: ContentAddressedArtifact
    attestation: ReceiptAttestation
    observation_fingerprint: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("observation_id")
    @classmethod
    def observation_id_is_canonical(cls, value: str) -> str:
        return _canonical_text(value, name="isolated-loader observation ID")

    @field_validator(
        "requirement_fingerprint",
        "implementation_scope_fingerprint",
        "loader_fingerprint",
        "observation_fingerprint",
    )
    @classmethod
    def hashes_are_canonical(cls, value: str) -> str:
        return _canonical_sha256(value)

    @field_validator("issued_at")
    @classmethod
    def issued_at_is_utc(cls, value: datetime) -> datetime:
        return _require_utc(value, name="isolated-loader observation time")

    @model_validator(mode="after")
    def observation_is_self_consistent(self) -> IsolatedLoadedInterfaceObservation:
        if self.loader_fingerprint != self.loader.fingerprint:
            raise ValueError("isolated-loader fingerprint does not match its exact identity")
        if (
            AdmissionVerifierCapability.LOADED_INTERFACE_VERIFICATION
            not in self.loader.capabilities
        ):
            raise ValueError("isolated loader lacks loaded-interface verification capability")
        attested_payload = self.model_dump(
            mode="python",
            exclude={"attestation", "observation_fingerprint"},
        )
        if self.attestation.attested_payload_fingerprint != canonical_fingerprint(attested_payload):
            raise ValueError("isolated-loader attestation binds a different observation")
        expected = canonical_fingerprint(
            self.model_dump(mode="python", exclude={"observation_fingerprint"})
        )
        if self.observation_fingerprint != expected:
            raise ValueError("isolated-loader observation fingerprint does not match its payload")
        return self


LoadedInterfaceReceipt.model_rebuild()


def require_valid_isolated_loader_observation(
    observation: IsolatedLoadedInterfaceObservation,
    *,
    trusted_loader: TrustedAdmissionVerifier,
) -> IsolatedLoadedInterfaceObservation:
    """Authenticate a persisted isolated-loader observation against its external trust root."""

    observation = IsolatedLoadedInterfaceObservation.model_validate(
        observation.model_dump(mode="python")
    )
    attested_payload = observation.model_dump(
        mode="python",
        exclude={"attestation", "observation_fingerprint"},
    )
    require_valid_canonical_attestation(
        attested_payload,
        observation.attestation,
        verifier_identity=observation.loader,
        trusted_verifier=trusted_loader,
        required_capability=AdmissionVerifierCapability.LOADED_INTERFACE_VERIFICATION,
    )
    return observation


def attest_isolated_loaded_interface_observation(
    *,
    observation_id: str,
    requirement: ImplementationReceiptRequirement,
    loaded_object: LoadedObjectIdentity,
    conformance: InterfaceConformanceObservation,
    trusted_loader: TrustedAdmissionVerifier,
    issued_at: datetime,
    isolation_evidence_artifact: ContentAddressedArtifact,
) -> IsolatedLoadedInterfaceObservation:
    """Authenticate a bounded observation produced outside the receipt-issuer process.

    This function never imports or executes implementation code.  The application must call it
    only after a separately isolated loader returns the supplied bounded identities/signatures.
    """

    requirement = ImplementationReceiptRequirement.model_validate(
        requirement.model_dump(mode="python")
    )
    loaded_object = LoadedObjectIdentity.model_validate(loaded_object.model_dump(mode="python"))
    conformance = InterfaceConformanceObservation.model_validate(
        conformance.model_dump(mode="python")
    )
    payload = {
        "schema_version": ADMISSION_RECEIPT_SCHEMA_VERSION,
        "observation_kind": "isolated_loaded_interface",
        "observation_id": observation_id,
        "requirement_fingerprint": requirement.fingerprint,
        "implementation_scope_fingerprint": requirement.implementation_scope_fingerprint,
        "implementation_code_artifact": requirement.implementation.code_artifact,
        "loaded_object": loaded_object,
        "conformance": conformance,
        "loader": trusted_loader.identity,
        "loader_fingerprint": trusted_loader.identity.fingerprint,
        "issued_at": issued_at,
        "isolation_evidence_artifact": isolation_evidence_artifact,
    }
    attestation = attest_canonical_payload(
        payload,
        trusted_verifier=trusted_loader,
        required_capability=AdmissionVerifierCapability.LOADED_INTERFACE_VERIFICATION,
    )
    observation_payload = {**payload, "attestation": attestation}
    return IsolatedLoadedInterfaceObservation.model_validate(
        {
            **observation_payload,
            "observation_fingerprint": canonical_fingerprint(observation_payload),
        }
    )


def issue_loaded_interface_receipt(
    *,
    receipt_id: str,
    requirement: ImplementationReceiptRequirement,
    isolated_observation: IsolatedLoadedInterfaceObservation,
    trusted_runtime_interface: TrustedRuntimeInterface,
    trusted_loader: TrustedAdmissionVerifier,
    trusted_verifier: TrustedAdmissionVerifier,
    issued_at: datetime,
    evidence_artifacts: Sequence[ContentAddressedArtifact],
) -> LoadedInterfaceReceipt:
    """Validate an authenticated isolated-loader observation and issue a receipt."""

    requirement = ImplementationReceiptRequirement.model_validate(
        requirement.model_dump(mode="python")
    )
    implementation = requirement.implementation
    if implementation.entrypoint is None:
        raise ValueError("loaded interface verification requires a declared Python entry point")
    if trusted_runtime_interface.declared_interface != implementation.interface:
        raise ValueError("trusted runtime interface does not match the implementation declaration")
    if trusted_runtime_interface.fingerprint != requirement.trusted_interface_fingerprint:
        raise ValueError("trusted runtime interface registry entry is stale or substituted")
    runtime_interface = trusted_runtime_interface.runtime_interface
    runtime_module, runtime_qualname = _runtime_interface_identity(runtime_interface)
    if (
        runtime_module != requirement.runtime_interface_module
        or runtime_qualname != requirement.runtime_interface_qualname
    ):
        raise ValueError("runtime interface object does not match the bound interface identity")
    required_signatures = _runtime_interface_members(runtime_interface)
    if required_signatures != requirement.required_member_signatures:
        raise ValueError("runtime interface signatures do not match the bound requirement")
    expected_interface_fingerprint = _interface_contract_fingerprint(
        implementation.interface,
        runtime_module,
        runtime_qualname,
        required_signatures,
    )
    if expected_interface_fingerprint != requirement.interface_contract_fingerprint:
        raise ValueError("runtime interface contract does not match its bound fingerprint")
    observation = IsolatedLoadedInterfaceObservation.model_validate(
        isolated_observation.model_dump(mode="python")
    )
    if observation.requirement_fingerprint != requirement.fingerprint:
        raise ValueError("isolated-loader observation is bound to a different requirement")
    if observation.implementation_scope_fingerprint != requirement.implementation_scope_fingerprint:
        raise ValueError("isolated-loader observation is bound to a stale implementation scope")
    if observation.implementation_code_artifact != implementation.code_artifact:
        raise ValueError("isolated-loader observation used different implementation code bytes")
    require_valid_isolated_loader_observation(
        observation,
        trusted_loader=trusted_loader,
    )
    loaded_identity = observation.loaded_object
    entrypoint_module, entrypoint_symbol = implementation.entrypoint.split(":", maxsplit=1)
    if loaded_identity.module != entrypoint_module or loaded_identity.qualname != entrypoint_symbol:
        raise ValueError("loaded object identity does not exactly match the declared entrypoint")
    conformance = observation.conformance
    if (
        conformance.declared_interface != implementation.interface
        or conformance.runtime_interface_module != runtime_module
        or conformance.runtime_interface_qualname != runtime_qualname
        or conformance.required_member_signatures != required_signatures
        or conformance.observed_matching_member_signatures != required_signatures
        or conformance.interface_contract_fingerprint != expected_interface_fingerprint
        or conformance.loaded_object_fingerprint != loaded_identity.fingerprint
    ):
        raise ValueError("isolated loaded object does not match the exact trusted interface")
    evidence = tuple(evidence_artifacts)
    payload = {
        "schema_version": ADMISSION_RECEIPT_SCHEMA_VERSION,
        "receipt_kind": "loaded_interface",
        "receipt_id": receipt_id,
        "requirement": requirement,
        "loaded_object": loaded_identity,
        "conformance": conformance,
        "isolated_loader_observation": observation,
        "execution_recheck_requirement": "reload_exact_bytes_and_reverify_before_execution",
        "verifier": trusted_verifier.identity,
        "verifier_fingerprint": trusted_verifier.identity.fingerprint,
        "issued_at": issued_at,
        "evidence_artifacts": evidence,
    }
    attestation = attest_canonical_payload(
        payload,
        trusted_verifier=trusted_verifier,
        required_capability=AdmissionVerifierCapability.LOADED_INTERFACE_VERIFICATION,
    )
    receipt_payload = {**payload, "attestation": attestation}
    return LoadedInterfaceReceipt.model_validate(
        {
            **receipt_payload,
            "receipt_fingerprint": canonical_fingerprint(receipt_payload),
        }
    )


@dataclass(frozen=True, slots=True)
class TrustedJITLoader:
    """Runtime-only application trust root that loads one exact immutable code snapshot.

    The callback is deliberately absent from every serialized contract.  Its verifier identity
    must be the same authenticated isolated loader that produced the admission observation.  A
    dishonest trusted loader remains inside the application's explicit trusted computing base;
    report submitters cannot replace it through a guard argument.
    """

    verifier: TrustedAdmissionVerifier
    load_exact: Callable[[LoadedInterfaceReceipt, bytes], object] = field(repr=False)

    def __post_init__(self) -> None:
        if (
            AdmissionVerifierCapability.LOADED_INTERFACE_VERIFICATION
            not in self.verifier.identity.capabilities
        ):
            raise ValueError("trusted JIT loader lacks loaded-interface verification capability")
        if not callable(self.load_exact):
            raise ValueError("trusted JIT loader must expose a callable exact-byte loader")

    @property
    def fingerprint(self) -> str:
        return self.verifier.identity.fingerprint


@dataclass(frozen=True, slots=True)
class VerifiedRuntimeHandle:
    """Non-serializable JIT authority over the exact object that may be invoked.

    A persisted receipt never contains this handle.  The execution worker must reacquire the
    declared code bytes, load them in its isolation boundary, and pass the resulting object to
    :func:`reverify_jit_loaded_interface` immediately before invocation.
    """

    loaded_object: object = field(repr=False)
    receipt_fingerprint: str
    requirement_fingerprint: str
    target_kind: ImplementationReceiptTargetKind
    target_id: str
    loaded_object_identity: LoadedObjectIdentity
    observed_code_sha256: str
    observed_code_byte_count: int
    conformance_fingerprint: str


def _required_member_names(signatures: tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(sorted(signature.split("|", maxsplit=1)[0] for signature in signatures))
    return _require_sorted_unique_text(
        names,
        name="required interface member names",
        allow_empty=False,
    )


def _require_concrete_loaded_members(
    loaded_object: object,
    member_names: tuple[str, ...],
) -> None:
    owner = loaded_object if inspect.isclass(loaded_object) else type(loaded_object)
    if inspect.isclass(owner) and inspect.isabstract(owner):
        raise ValueError("loaded implementation class must not remain abstract")
    if not inspect.isclass(owner):
        raise ValueError("loaded implementation does not expose class-owned interface members")
    for member_name in member_names:
        declaring_owner: type[object] | None = None
        descriptor: object | None = None
        for ancestor in owner.__mro__:
            if member_name in ancestor.__dict__:
                declaring_owner = ancestor
                descriptor = ancestor.__dict__[member_name]
                break
        if declaring_owner is None:
            raise ValueError(f"loaded object is missing required member {member_name!r}")
        is_protocol_stub = bool(getattr(declaring_owner, "_is_protocol", False))
        is_abstract_member = bool(getattr(descriptor, "__isabstractmethod__", False))
        if isinstance(descriptor, (classmethod, staticmethod)):
            is_abstract_member = is_abstract_member or bool(
                getattr(descriptor.__func__, "__isabstractmethod__", False)
            )
        if isinstance(descriptor, property):
            is_abstract_member = is_abstract_member or bool(
                descriptor.fget is not None
                and getattr(descriptor.fget, "__isabstractmethod__", False)
            )
        if is_protocol_stub or is_abstract_member:
            raise ValueError(
                f"loaded object inherits an unimplemented interface member {member_name!r}"
            )


def reverify_jit_loaded_interface(
    receipt: LoadedInterfaceReceipt,
    *,
    resolved_code_content: ObservedByteSource,
    trusted_runtime_interface: TrustedRuntimeInterface,
    trusted_jit_loader: TrustedJITLoader,
) -> VerifiedRuntimeHandle:
    """Derive byte and interface observations from the real JIT object, then return authority.

    This function performs no import or source execution and carries no verifier secret.  Call it
    inside the isolated execution worker after a trusted parent has authenticated ``receipt``.
    Invocation must use ``handle.loaded_object`` itself so the checked object cannot be swapped.
    """

    receipt = LoadedInterfaceReceipt.model_validate(receipt.model_dump(mode="python"))
    requirement = receipt.requirement
    implementation = requirement.implementation
    if implementation.entrypoint is None:
        raise ValueError("JIT interface verification requires an executable entrypoint")
    if trusted_runtime_interface.fingerprint != requirement.trusted_interface_fingerprint:
        raise ValueError("JIT runtime interface registry entry is stale or substituted")
    if trusted_runtime_interface.interface_artifact != requirement.interface_artifact:
        raise ValueError("JIT runtime interface artifact differs from the admitted requirement")

    loader_observation = receipt.isolated_loader_observation
    if trusted_jit_loader.verifier.identity != loader_observation.loader:
        raise ValueError("JIT loader differs from the authenticated isolated admission loader")
    if trusted_jit_loader.fingerprint != loader_observation.loader_fingerprint:
        raise ValueError("JIT loader fingerprint differs from the admission observation")
    if trusted_jit_loader.verifier.key_id != loader_observation.attestation.key_id:
        raise ValueError("JIT loader key differs from the authenticated admission observation")
    require_valid_isolated_loader_observation(
        loader_observation,
        trusted_loader=trusted_jit_loader.verifier,
    )

    sealed_code = _seal_byte_source(
        resolved_code_content,
        expected_byte_count=implementation.code_artifact.byte_count,
    )
    observed_sha256 = sha256(sealed_code).hexdigest()
    observed_byte_count = len(sealed_code)
    if (
        observed_sha256 != implementation.code_artifact.sha256
        or observed_byte_count != implementation.code_artifact.byte_count
    ):
        raise ValueError("JIT code bytes do not match the admitted implementation artifact")

    loaded_object = trusted_jit_loader.load_exact(receipt, sealed_code)

    loaded_identity = _loaded_object_identity(
        loaded_object,
        entrypoint=implementation.entrypoint,
        loaded_code_sha256=observed_sha256,
    )
    entrypoint_module, entrypoint_symbol = implementation.entrypoint.split(":", maxsplit=1)
    if loaded_identity.module != entrypoint_module or loaded_identity.qualname != entrypoint_symbol:
        raise ValueError("JIT loaded object identity does not match the admitted entrypoint")
    if loaded_identity != receipt.loaded_object:
        raise ValueError("JIT loaded object identity differs from the isolated admission receipt")

    runtime_interface = trusted_runtime_interface.runtime_interface
    runtime_module, runtime_qualname = _runtime_interface_identity(runtime_interface)
    required_signatures = _runtime_interface_members(runtime_interface)
    if (
        runtime_module != requirement.runtime_interface_module
        or runtime_qualname != requirement.runtime_interface_qualname
        or required_signatures != requirement.required_member_signatures
    ):
        raise ValueError("JIT trusted interface differs from the admitted contract")
    verification_method = _verification_method(runtime_interface, loaded_object)
    if verification_method is not receipt.conformance.verification_method:
        raise ValueError("JIT interface verification method differs from admission")
    _require_nominal_conformance(loaded_object, runtime_interface, verification_method)
    member_names = _required_member_names(required_signatures)
    _require_concrete_loaded_members(loaded_object, member_names)
    observed_signatures = tuple(
        sorted(_member_signature(loaded_object, member_name) for member_name in member_names)
    )
    if observed_signatures != required_signatures:
        raise ValueError("JIT loaded object signatures differ from the admitted contract")
    observed_contract_fingerprint = _interface_contract_fingerprint(
        implementation.interface,
        runtime_module,
        runtime_qualname,
        observed_signatures,
    )
    if observed_contract_fingerprint != requirement.interface_contract_fingerprint:
        raise ValueError("JIT loaded object contract fingerprint differs from admission")
    return VerifiedRuntimeHandle(
        loaded_object=loaded_object,
        receipt_fingerprint=receipt.receipt_fingerprint,
        requirement_fingerprint=requirement.fingerprint,
        target_kind=requirement.target_kind,
        target_id=requirement.target_id,
        loaded_object_identity=loaded_identity,
        observed_code_sha256=observed_sha256,
        observed_code_byte_count=observed_byte_count,
        conformance_fingerprint=observed_contract_fingerprint,
    )


def consumed_artifacts_for_bundle(
    bundle: BiologicalModelBundleContract,
    *,
    execution_source_selection: ExecutionSourceSelectionReceipt,
    runtime_interfaces: Mapping[str, TrustedRuntimeInterface],
    additional_artifacts: Sequence[ContentAddressedArtifact] = (),
) -> tuple[AdmissionArtifactReference, ...]:
    """Return each unique exact artifact declaration consumed by bundle admission once.

    Execution sources come only from an authenticated workflow selection; interface artifacts
    come only from the application-owned runtime registry.  Review-only manifest sources are not
    inferred, and neither category can be omitted by a report submitter.
    """

    bundle = BiologicalModelBundleContract.model_validate(bundle.model_dump(mode="python"))
    execution_source_selection = ExecutionSourceSelectionReceipt.model_validate(
        execution_source_selection.model_dump(mode="python")
    )
    if execution_source_selection.bundle_fingerprint != bundle.fingerprint:
        raise ValueError("execution-source selection is bound to a different bundle")
    implementation_requirements = implementation_requirements_for_bundle(
        bundle,
        runtime_interfaces=runtime_interfaces,
    )

    candidates: list[ArtifactDeclaration] = [
        bundle.query.artifact,
        bundle.benchmark.artifact,
        bundle.support_envelope.artifact,
    ]
    if bundle.model_artifact is not None:
        candidates.append(bundle.model_artifact)
    if bundle.training_run is not None:
        candidates.append(bundle.training_run.artifact)
    candidates.extend(reference.artifact for reference in bundle.validation_evidence)
    candidates.extend(
        binding.implementation.code_artifact
        for binding in bundle.ports
        if binding.disposition is PortDisposition.PROVIDED and binding.implementation is not None
    )
    candidates.extend(
        binding.implementation.code_artifact for binding in bundle.operation_implementations
    )
    candidates.extend(execution_source_selection.workflow_resolution_artifacts)
    candidates.extend(execution_source_selection.sources)
    candidates.extend(requirement.interface_artifact for requirement in implementation_requirements)
    candidates.extend(additional_artifacts)

    by_identity: dict[str, AdmissionArtifactReference] = {}
    by_id: dict[str, str] = {}
    for declaration in candidates:
        artifact = admission_artifact_reference(declaration)
        identity = _artifact_identity(artifact)
        previous_identity = by_id.get(artifact.target_key)
        if previous_identity is not None and previous_identity != identity:
            raise ValueError("one artifact ID cannot name conflicting declarations")
        by_id[artifact.target_key] = identity
        by_identity.setdefault(identity, artifact)
    artifacts = tuple(sorted(by_identity.values(), key=lambda item: item.target_key))
    return _require_canonical_artifacts(artifacts, name="consumed bundle artifacts")


def implementation_requirements_for_bundle(
    bundle: BiologicalModelBundleContract,
    *,
    runtime_interfaces: Mapping[str, TrustedRuntimeInterface],
) -> tuple[ImplementationReceiptRequirement, ...]:
    """Derive exact receipt requirements for every provided port and runtime operation."""

    bundle = BiologicalModelBundleContract.model_validate(bundle.model_dump(mode="python"))
    requirements: list[ImplementationReceiptRequirement] = []

    def requirement_for(
        *,
        target_kind: ImplementationReceiptTargetKind,
        target_id: str,
        implementation: PortImplementationBinding,
    ) -> ImplementationReceiptRequirement:
        trusted_interface = runtime_interfaces.get(implementation.interface)
        if trusted_interface is None:
            raise ValueError(
                f"no trusted runtime interface is registered for {implementation.interface!r}"
            )
        if trusted_interface.declared_interface != implementation.interface:
            raise ValueError("trusted runtime interface registry key/declaration mismatch")
        runtime_module, runtime_qualname = _runtime_interface_identity(
            trusted_interface.runtime_interface
        )
        member_signatures = _runtime_interface_members(trusted_interface.runtime_interface)
        return ImplementationReceiptRequirement(
            target_kind=target_kind,
            target_id=target_id,
            bundle_fingerprint=bundle.fingerprint,
            implementation_scope_fingerprint=bundle.implementation_scope_fingerprint,
            implementation=implementation,
            interface_artifact=trusted_interface.interface_artifact,
            trusted_interface_fingerprint=trusted_interface.fingerprint,
            runtime_interface_module=runtime_module,
            runtime_interface_qualname=runtime_qualname,
            required_member_signatures=member_signatures,
            interface_contract_fingerprint=_interface_contract_fingerprint(
                implementation.interface,
                runtime_module,
                runtime_qualname,
                member_signatures,
            ),
        )

    for port_binding in bundle.ports:
        if port_binding.disposition is not PortDisposition.PROVIDED:
            continue
        if port_binding.implementation is None:
            raise ValueError("a provided port has no implementation declaration")
        requirements.append(
            requirement_for(
                target_kind=ImplementationReceiptTargetKind.PORT,
                target_id=port_binding.port.value,
                implementation=port_binding.implementation,
            )
        )
    for operation_binding in bundle.operation_implementations:
        requirements.append(
            requirement_for(
                target_kind=ImplementationReceiptTargetKind.RUNTIME_OPERATION,
                target_id=operation_binding.operation.value,
                implementation=operation_binding.implementation,
            )
        )
    requirements.sort(key=lambda item: item.target_key)
    target_keys = tuple(requirement.target_key for requirement in requirements)
    _require_sorted_unique_text(target_keys, name="implementation receipt target keys")
    return tuple(requirements)


def artifact_coverage_fingerprint(
    artifacts: Sequence[AdmissionArtifactReference],
) -> str:
    """Fingerprint an already canonical exact artifact coverage requirement."""

    required = tuple(artifacts)
    _require_canonical_artifacts(required, name="artifact coverage")
    return canonical_fingerprint([artifact.model_dump(mode="json") for artifact in required])


def interface_coverage_fingerprint(
    requirements: Sequence[ImplementationReceiptRequirement],
) -> str:
    """Fingerprint an already canonical exact interface coverage requirement."""

    required = tuple(requirements)
    keys = tuple(requirement.target_key for requirement in required)
    _require_sorted_unique_text(keys, name="interface coverage target keys")
    return canonical_fingerprint([requirement.model_dump(mode="json") for requirement in required])


class AdmissionReceiptBatchReport(AdmissionReceiptModel):
    """Complete one-to-one artifact and interface receipt coverage for one bundle scope."""

    schema_version: AdmissionReceiptSchemaVersion = ADMISSION_RECEIPT_SCHEMA_VERSION
    report_kind: Literal["complete_admission_receipt_batch"] = "complete_admission_receipt_batch"
    batch_id: str = Field(min_length=1)
    bundle_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    implementation_scope_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    execution_source_selection: ExecutionSourceSelectionReceipt
    required_artifacts: tuple[AdmissionArtifactReference, ...] = Field(min_length=1)
    artifact_receipts: tuple[ArtifactResolutionReceipt, ...] = Field(min_length=1)
    required_interfaces: tuple[ImplementationReceiptRequirement, ...] = ()
    interface_receipts: tuple[LoadedInterfaceReceipt, ...] = ()
    artifact_coverage_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    interface_coverage_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    issued_at: datetime
    evidence_artifacts: tuple[ContentAddressedArtifact, ...] = Field(
        min_length=1,
        description=(
            "Non-authoritative batch audit-log references; only the closed required/receipt "
            "sets above contribute admission coverage."
        ),
    )
    report_fingerprint: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("batch_id")
    @classmethod
    def batch_id_is_canonical(cls, value: str) -> str:
        return _canonical_text(value, name="admission receipt batch ID")

    @field_validator(
        "bundle_fingerprint",
        "implementation_scope_fingerprint",
        "artifact_coverage_fingerprint",
        "interface_coverage_fingerprint",
        "report_fingerprint",
    )
    @classmethod
    def fingerprints_are_canonical(cls, value: str) -> str:
        return _canonical_sha256(value)

    @field_validator("issued_at")
    @classmethod
    def issued_at_is_utc(cls, value: datetime) -> datetime:
        return _require_utc(value, name="admission receipt batch issuance time")

    @field_validator("required_artifacts")
    @classmethod
    def required_artifacts_are_canonical(
        cls, value: tuple[AdmissionArtifactReference, ...]
    ) -> tuple[AdmissionArtifactReference, ...]:
        return _require_canonical_artifacts(value, name="required admission artifacts")

    @field_validator("evidence_artifacts")
    @classmethod
    def evidence_is_canonical(
        cls, value: tuple[ContentAddressedArtifact, ...]
    ) -> tuple[ContentAddressedArtifact, ...]:
        return _require_canonical_evidence_artifacts(value, name="admission batch evidence")

    @model_validator(mode="after")
    def coverage_is_complete_and_exact(self) -> AdmissionReceiptBatchReport:
        if self.execution_source_selection.bundle_fingerprint != self.bundle_fingerprint:
            raise ValueError("execution-source selection must bind the report's exact bundle")
        if self.execution_source_selection.issued_at > self.issued_at:
            raise ValueError("execution-source selection cannot postdate its batch report")
        required_interface_keys = tuple(item.target_key for item in self.required_interfaces)
        _require_sorted_unique_text(
            required_interface_keys,
            name="required interface target keys",
        )
        artifact_receipt_ids = tuple(receipt.receipt_id for receipt in self.artifact_receipts)
        _require_sorted_unique_text(
            artifact_receipt_ids,
            name="artifact receipt IDs",
            allow_empty=False,
        )
        interface_receipt_ids = tuple(receipt.receipt_id for receipt in self.interface_receipts)
        _require_sorted_unique_text(interface_receipt_ids, name="interface receipt IDs")
        all_receipt_ids = artifact_receipt_ids + interface_receipt_ids
        if len(all_receipt_ids) != len(set(all_receipt_ids)):
            raise ValueError("receipt IDs must be unique across the complete batch")

        required_artifact_identities = tuple(
            _artifact_identity(artifact) for artifact in self.required_artifacts
        )
        receipt_artifact_identities = tuple(
            _artifact_identity(receipt.artifact) for receipt in self.artifact_receipts
        )
        if len(receipt_artifact_identities) != len(set(receipt_artifact_identities)):
            raise ValueError("an exact artifact may have only one resolution receipt per batch")
        if set(receipt_artifact_identities) != set(required_artifact_identities):
            raise ValueError("artifact receipts must cover every required exact artifact once")
        if len(receipt_artifact_identities) != len(required_artifact_identities):
            raise ValueError("artifact receipt coverage must be one-to-one")
        selected_artifacts: tuple[ArtifactDeclaration, ...] = (
            tuple(self.execution_source_selection.workflow_resolution_artifacts)
            + tuple(self.execution_source_selection.sources)
            + tuple(requirement.interface_artifact for requirement in self.required_interfaces)
        )
        selected_identities = {
            _artifact_identity(admission_artifact_reference(artifact))
            for artifact in selected_artifacts
        }
        if not selected_identities.issubset(set(required_artifact_identities)):
            raise ValueError(
                "required artifacts must include every selected source, workflow resolution, "
                "and trusted interface artifact"
            )
        required_reference_ids = {artifact.reference_id for artifact in self.required_artifacts}
        if required_reference_ids.intersection(
            evidence.artifact_id for evidence in self.evidence_artifacts
        ):
            raise ValueError(
                "non-authoritative batch evidence must be distinct from required artifacts"
            )

        required_by_target = {
            requirement.target_key: requirement.fingerprint
            for requirement in self.required_interfaces
        }
        receipt_target_keys = tuple(
            receipt.requirement.target_key for receipt in self.interface_receipts
        )
        if len(receipt_target_keys) != len(set(receipt_target_keys)):
            raise ValueError("an implementation target may have only one interface receipt")
        receipt_by_target = {
            receipt.requirement.target_key: receipt.requirement.fingerprint
            for receipt in self.interface_receipts
        }
        if receipt_by_target != required_by_target:
            raise ValueError("interface receipts must cover every exact implementation target once")

        if any(
            requirement.bundle_fingerprint != self.bundle_fingerprint
            for requirement in self.required_interfaces
        ):
            raise ValueError("interface requirements must bind the report's exact bundle")
        if any(
            requirement.implementation_scope_fingerprint != self.implementation_scope_fingerprint
            for requirement in self.required_interfaces
        ):
            raise ValueError("interface requirements must bind the report's implementation scope")
        if any(receipt.issued_at > self.issued_at for receipt in self.artifact_receipts):
            raise ValueError("artifact receipts cannot postdate their batch report")
        if any(receipt.issued_at > self.issued_at for receipt in self.interface_receipts):
            raise ValueError("interface receipts cannot postdate their batch report")

        expected_artifact_coverage = artifact_coverage_fingerprint(self.required_artifacts)
        if self.artifact_coverage_fingerprint != expected_artifact_coverage:
            raise ValueError("artifact coverage fingerprint does not match its exact requirements")
        expected_interface_coverage = interface_coverage_fingerprint(self.required_interfaces)
        if self.interface_coverage_fingerprint != expected_interface_coverage:
            raise ValueError("interface coverage fingerprint does not match its exact requirements")
        expected_report = canonical_fingerprint(
            self.model_dump(mode="python", exclude={"report_fingerprint"})
        )
        if self.report_fingerprint != expected_report:
            raise ValueError("admission receipt batch fingerprint does not match its payload")
        return self

    def artifact_receipt_for(self, artifact: ArtifactDeclaration) -> ArtifactResolutionReceipt:
        """Return the single exact receipt for ``artifact`` or fail closed."""

        identity = _artifact_identity(admission_artifact_reference(artifact))
        matches = tuple(
            receipt
            for receipt in self.artifact_receipts
            if _artifact_identity(receipt.artifact) == identity
        )
        if len(matches) != 1:
            raise ValueError("batch does not contain exactly one receipt for the artifact")
        return matches[0]

    def interface_receipt_for(
        self,
        *,
        target_kind: ImplementationReceiptTargetKind,
        target_id: str,
    ) -> LoadedInterfaceReceipt:
        """Return the single exact receipt for one implementation target or fail closed."""

        key = f"{target_kind.value}:{target_id}"
        matches = tuple(
            receipt for receipt in self.interface_receipts if receipt.requirement.target_key == key
        )
        if len(matches) != 1:
            raise ValueError("batch does not contain exactly one receipt for the interface target")
        return matches[0]


def build_admission_receipt_batch_report(
    *,
    batch_id: str,
    bundle: BiologicalModelBundleContract,
    artifact_receipts: Sequence[ArtifactResolutionReceipt],
    interface_receipts: Sequence[LoadedInterfaceReceipt],
    issued_at: datetime,
    evidence_artifacts: Sequence[ContentAddressedArtifact],
    runtime_interfaces: Mapping[str, TrustedRuntimeInterface],
    execution_source_selection: ExecutionSourceSelectionReceipt,
    trusted_execution_source_selector: TrustedAdmissionVerifier,
    additional_required_artifacts: Sequence[ContentAddressedArtifact] = (),
) -> AdmissionReceiptBatchReport:
    """Derive exact requirements from a bundle and construct a complete batch report."""

    bundle = BiologicalModelBundleContract.model_validate(bundle.model_dump(mode="python"))
    selection = require_valid_execution_source_selection(
        execution_source_selection,
        trusted_selector=trusted_execution_source_selector,
    )
    required_artifacts = consumed_artifacts_for_bundle(
        bundle,
        execution_source_selection=selection,
        runtime_interfaces=runtime_interfaces,
        additional_artifacts=additional_required_artifacts,
    )
    required_interfaces = implementation_requirements_for_bundle(
        bundle,
        runtime_interfaces=runtime_interfaces,
    )
    sorted_artifact_receipts = tuple(
        sorted(artifact_receipts, key=lambda receipt: receipt.receipt_id)
    )
    sorted_interface_receipts = tuple(
        sorted(interface_receipts, key=lambda receipt: receipt.receipt_id)
    )
    evidence = tuple(evidence_artifacts)
    payload = {
        "schema_version": ADMISSION_RECEIPT_SCHEMA_VERSION,
        "report_kind": "complete_admission_receipt_batch",
        "batch_id": batch_id,
        "bundle_fingerprint": bundle.fingerprint,
        "implementation_scope_fingerprint": bundle.implementation_scope_fingerprint,
        "execution_source_selection": selection,
        "required_artifacts": required_artifacts,
        "artifact_receipts": sorted_artifact_receipts,
        "required_interfaces": required_interfaces,
        "interface_receipts": sorted_interface_receipts,
        "artifact_coverage_fingerprint": artifact_coverage_fingerprint(required_artifacts),
        "interface_coverage_fingerprint": interface_coverage_fingerprint(required_interfaces),
        "issued_at": issued_at,
        "evidence_artifacts": evidence,
    }
    return AdmissionReceiptBatchReport.model_validate(
        {**payload, "report_fingerprint": canonical_fingerprint(payload)}
    )


def require_exact_receipt_batch_coverage(
    bundle: BiologicalModelBundleContract,
    report: AdmissionReceiptBatchReport,
    *,
    trusted_verifiers: Sequence[TrustedAdmissionVerifier],
    runtime_interfaces: Mapping[str, TrustedRuntimeInterface],
    additional_required_artifacts: Sequence[ContentAddressedArtifact] = (),
) -> AdmissionReceiptBatchReport:
    """Rebind a persisted report to a source bundle and fail on any stale or missing receipt."""

    bundle = BiologicalModelBundleContract.model_validate(bundle.model_dump(mode="python"))
    report = AdmissionReceiptBatchReport.model_validate(report.model_dump(mode="python"))
    if report.implementation_scope_fingerprint != bundle.implementation_scope_fingerprint:
        raise ValueError("admission receipt report is bound to a stale implementation scope")
    if report.bundle_fingerprint != bundle.fingerprint:
        raise ValueError("admission receipt report is bound to a different bundle fingerprint")
    report = require_trusted_receipt_verifiers(
        report,
        trusted_verifiers=trusted_verifiers,
    )
    expected_artifacts = consumed_artifacts_for_bundle(
        bundle,
        execution_source_selection=report.execution_source_selection,
        runtime_interfaces=runtime_interfaces,
        additional_artifacts=additional_required_artifacts,
    )
    if report.required_artifacts != expected_artifacts:
        raise ValueError("admission receipt report does not cover the bundle's exact artifacts")
    expected_interfaces = implementation_requirements_for_bundle(
        bundle,
        runtime_interfaces=runtime_interfaces,
    )
    if report.required_interfaces != expected_interfaces:
        raise ValueError("admission receipt report does not cover the bundle's exact interfaces")
    return report


def require_trusted_receipt_verifiers(
    report: AdmissionReceiptBatchReport,
    *,
    trusted_verifiers: Sequence[TrustedAdmissionVerifier],
) -> AdmissionReceiptBatchReport:
    """Authenticate every receipt against an external, non-serialized trust root."""

    report = AdmissionReceiptBatchReport.model_validate(report.model_dump(mode="python"))
    if not trusted_verifiers:
        raise ValueError("admission receipt verification requires an explicit verifier trust root")
    trusted_by_key: dict[tuple[str, str], TrustedAdmissionVerifier] = {}
    for verifier in trusted_verifiers:
        identity = AdmissionVerifierIdentity.model_validate(
            verifier.identity.model_dump(mode="python")
        )
        validated_verifier = TrustedAdmissionVerifier(
            identity=identity,
            key_id=verifier.key_id,
            secret=verifier.secret,
        )
        key = (identity.fingerprint, validated_verifier.key_id)
        if key in trusted_by_key:
            raise ValueError("trusted verifier identity/key pairs must be unique")
        trusted_by_key[key] = validated_verifier

    def trusted_root_for(
        identity: AdmissionVerifierIdentity,
        identity_fingerprint: str,
        attestation: ReceiptAttestation,
    ) -> TrustedAdmissionVerifier:
        key = (identity_fingerprint, attestation.key_id)
        trusted = trusted_by_key.get(key)
        if trusted is None or trusted.identity != identity:
            raise ValueError("receipt verifier is absent from the exact external trust root")
        return trusted

    selection = report.execution_source_selection
    trusted_selector = trusted_root_for(
        selection.selector,
        selection.selector_fingerprint,
        selection.attestation,
    )
    require_valid_execution_source_selection(
        selection,
        trusted_selector=trusted_selector,
    )

    for artifact_receipt in report.artifact_receipts:
        trusted = trusted_root_for(
            artifact_receipt.verifier,
            artifact_receipt.verifier_fingerprint,
            artifact_receipt.attestation,
        )
        attested_payload = artifact_receipt.model_dump(
            mode="python",
            exclude={"attestation", "receipt_fingerprint"},
        )
        require_valid_canonical_attestation(
            attested_payload,
            artifact_receipt.attestation,
            verifier_identity=artifact_receipt.verifier,
            trusted_verifier=trusted,
            required_capability=AdmissionVerifierCapability.ARTIFACT_BYTE_RESOLUTION,
        )
    for interface_receipt in report.interface_receipts:
        trusted = trusted_root_for(
            interface_receipt.verifier,
            interface_receipt.verifier_fingerprint,
            interface_receipt.attestation,
        )
        attested_payload = interface_receipt.model_dump(
            mode="python",
            exclude={"attestation", "receipt_fingerprint"},
        )
        require_valid_canonical_attestation(
            attested_payload,
            interface_receipt.attestation,
            verifier_identity=interface_receipt.verifier,
            trusted_verifier=trusted,
            required_capability=AdmissionVerifierCapability.LOADED_INTERFACE_VERIFICATION,
        )
        observation = interface_receipt.isolated_loader_observation
        trusted_loader = trusted_root_for(
            observation.loader,
            observation.loader_fingerprint,
            observation.attestation,
        )
        require_valid_isolated_loader_observation(
            observation,
            trusted_loader=trusted_loader,
        )
    return report


__all__ = [
    "ADMISSION_RECEIPT_SCHEMA_VERSION",
    "AdmissionArtifactKind",
    "AdmissionArtifactReference",
    "AdmissionReceiptBatchReport",
    "AdmissionVerifierCapability",
    "AdmissionVerifierIdentity",
    "ArtifactByteObservation",
    "ArtifactDeclaration",
    "ArtifactResolutionReceipt",
    "ExecutionSourceSelectionReceipt",
    "ImplementationReceiptRequirement",
    "ImplementationReceiptTargetKind",
    "InterfaceConformanceObservation",
    "InterfaceVerificationMethod",
    "IsolatedLoadedInterfaceObservation",
    "LoadedInterfaceReceipt",
    "LoadedObjectIdentity",
    "LoadedObjectKind",
    "ObservedByteSource",
    "ReceiptAttestation",
    "TrustedAdmissionVerifier",
    "TrustedJITLoader",
    "TrustedRuntimeInterface",
    "VerifiedRuntimeHandle",
    "admission_artifact_reference",
    "artifact_coverage_fingerprint",
    "attest_canonical_payload",
    "attest_isolated_loaded_interface_observation",
    "build_admission_receipt_batch_report",
    "consumed_artifacts_for_bundle",
    "implementation_requirements_for_bundle",
    "interface_coverage_fingerprint",
    "issue_artifact_resolution_receipt",
    "issue_execution_source_selection_receipt",
    "issue_loaded_interface_receipt",
    "require_exact_receipt_batch_coverage",
    "require_trusted_receipt_verifiers",
    "require_valid_canonical_attestation",
    "require_valid_execution_source_selection",
    "require_valid_isolated_loader_observation",
    "reverify_jit_loaded_interface",
]
