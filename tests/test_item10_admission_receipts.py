"""Adversarial tests for exact-byte and loaded-interface admission receipts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from typing import Protocol

import pytest
from pydantic import ValidationError

from cellstate.backends.admission import (
    AdmissionArtifactKind,
    AdmissionReceiptBatchReport,
    AdmissionVerifierCapability,
    AdmissionVerifierIdentity,
    ArtifactResolutionReceipt,
    ImplementationReceiptRequirement,
    InterfaceConformanceObservation,
    InterfaceVerificationMethod,
    LoadedObjectIdentity,
    LoadedObjectKind,
    ReceiptAttestation,
    TrustedAdmissionVerifier,
    TrustedJITLoader,
    TrustedRuntimeInterface,
    attest_canonical_payload,
    attest_isolated_loaded_interface_observation,
    build_admission_receipt_batch_report,
    consumed_artifacts_for_bundle,
    implementation_requirements_for_bundle,
    issue_artifact_resolution_receipt,
    issue_execution_source_selection_receipt,
    issue_loaded_interface_receipt,
    require_exact_receipt_batch_coverage,
    require_valid_canonical_attestation,
    require_valid_execution_source_selection,
    reverify_jit_loaded_interface,
)
from cellstate.backends.contracts import (
    BiologicalModelBundleContract,
    BiologicalStagePort,
    BundleContractKind,
    BundleContractReference,
    ModelOperation,
    ModelOperationImplementationBinding,
    ModelPortBinding,
    PortDisposition,
    PortImplementationBinding,
    PortImplementationKind,
)
from cellstate.data import ContentAddressedArtifact, SourceArtifact, SourceKind
from cellstate.domain.common import canonical_fingerprint

NOW = datetime(2026, 8, 10, 16, 0, tzinfo=UTC)
INTERFACE_NAME = f"{__name__}.ExampleRuntimeInterface"
ABSTRACT_INTERFACE_NAME = f"{__name__}.AbstractRuntimeInterface"
ENTRYPOINT = "cellstate_receipt_test_runtime:GoodImplementation"
CODE_BYTES = b"""from __future__ import annotations

class GoodImplementation:
    @property
    def descriptor(self) -> str:
        return "example"

    def run(self, value: int, *, scale: int = 1) -> str:
        return str(value * scale)
"""
INTERFACE_BYTES = b"trusted ExampleRuntimeInterface contract v1"
WORKFLOW_BYTES = b"typed assessment and permission resolution v1"
SOURCE_BYTES = b"real-h5ad-bytes"


class ExampleRuntimeInterface(Protocol):
    @property
    def descriptor(self) -> str: ...

    def run(self, value: int, *, scale: int = 1) -> str: ...


class GoodImplementation:
    @property
    def descriptor(self) -> str:
        return "example"

    def run(self, value: int, *, scale: int = 1) -> str:
        return str(value * scale)


class StaticMethodSubstitution:
    @property
    def descriptor(self) -> str:
        return "example"

    @staticmethod
    def run(value: int, *, scale: int = 1) -> str:
        return str(value * scale)


class InheritedProtocolStub(ExampleRuntimeInterface):
    pass


class AbstractRuntimeInterface(ABC):
    @property
    @abstractmethod
    def descriptor(self) -> str: ...

    @abstractmethod
    def run(self, value: int, *, scale: int = 1) -> str: ...


class NominalABCImplementation(AbstractRuntimeInterface):
    @property
    def descriptor(self) -> str:
        return "example"

    def run(self, value: int, *, scale: int = 1) -> str:
        return str(value * scale)


class SignatureOnlyABCImpostor:
    @property
    def descriptor(self) -> str:
        return "example"

    def run(self, value: int, *, scale: int = 1) -> str:
        return str(value * scale)


for implementation_class in (
    GoodImplementation,
    StaticMethodSubstitution,
    InheritedProtocolStub,
    NominalABCImplementation,
    SignatureOnlyABCImpostor,
):
    implementation_class.__module__ = "cellstate_receipt_test_runtime"
    implementation_class.__qualname__ = "GoodImplementation"


def _content_artifact(artifact_id: str, content: bytes) -> ContentAddressedArtifact:
    return ContentAddressedArtifact(
        artifact_id=artifact_id,
        uri=f"https://example.invalid/artifacts/{artifact_id}",
        sha256=sha256(content).hexdigest(),
        byte_count=len(content),
        media_type="application/octet-stream",
    )


def _source_artifact(content: bytes = SOURCE_BYTES) -> SourceArtifact:
    return SourceArtifact(
        source_id="sciplex3-k562-corrected-h5ad",
        kind=SourceKind.PROCESSED,
        uri="https://example.invalid/data/sciplex3-k562-corrected.h5ad",
        sha256=sha256(content).hexdigest(),
        media_type="application/x-hdf5",
        accession="GSE139944",
        release="2020-04-30",
        parent_study_accession="GSE139944",
        parent_study_release="2020-04-30",
        byte_count=len(content),
        retrieved_at=datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
    )


def _reference(name: str, content: bytes) -> BundleContractReference:
    return BundleContractReference(
        contract_id=name,
        contract_version="0.1.0",
        artifact=_content_artifact(name, content),
    )


def _implementation(code: bytes = CODE_BYTES) -> PortImplementationBinding:
    return PortImplementationBinding(
        implementation_id="cellstate.test.example-runtime",
        implementation_version="0.1.0",
        interface=INTERFACE_NAME,
        kind=PortImplementationKind.PYTHON_ENTRY_POINT,
        code_artifact=_content_artifact("example-runtime-code", code),
        entrypoint=ENTRYPOINT,
    )


def _bundle(code: bytes = CODE_BYTES) -> BiologicalModelBundleContract:
    implementation = _implementation(code)
    ports = []
    for port in BiologicalStagePort:
        if port is BiologicalStagePort.ACTION_CONTEXT_ENCODER:
            ports.append(
                ModelPortBinding(
                    port=port,
                    disposition=PortDisposition.PROVIDED,
                    implementation=implementation,
                    rationale=("Exact test implementation.",),
                )
            )
        else:
            ports.append(
                ModelPortBinding(
                    port=port,
                    disposition=PortDisposition.NOT_APPLICABLE,
                    rationale=("Outside the exact test scope.",),
                )
            )
    return BiologicalModelBundleContract(
        bundle_id="admission-receipt-test-bundle",
        bundle_version="0.1.0",
        bundle_kind=BundleContractKind.BIOLOGICAL_MODEL_BUNDLE,
        description="Exact test bundle for receipt-boundary verification.",
        query=_reference("receipt-test-query", b"query"),
        benchmark=_reference("receipt-test-benchmark", b"benchmark"),
        support_envelope=_reference("receipt-test-envelope", b"envelope"),
        ports=tuple(sorted(ports, key=lambda binding: binding.port.value)),
        operation_implementations=(
            ModelOperationImplementationBinding(
                operation=ModelOperation.ESTIMATE_CELL_STATE,
                implementation=implementation,
                rationale=("Exact test runtime operation.",),
            ),
        ),
    )


def _verifier(
    capabilities: tuple[AdmissionVerifierCapability, ...] | None = None,
) -> AdmissionVerifierIdentity:
    return AdmissionVerifierIdentity(
        verifier_id="cellstate.admission.verifier",
        verifier_version="0.1.0",
        code_artifact=_content_artifact("admission-verifier-code", b"verifier-code"),
        entrypoint="cellstate.backends.admission:admission_verifier",
        runtime="cpython-3.11",
        capabilities=capabilities
        or (
            AdmissionVerifierCapability.ARTIFACT_BYTE_RESOLUTION,
            AdmissionVerifierCapability.EXECUTION_SOURCE_SELECTION,
            AdmissionVerifierCapability.LOADED_INTERFACE_VERIFICATION,
            AdmissionVerifierCapability.VALIDATION_RESULT_SEMANTICS,
        ),
    )


def _trusted_verifier(
    *,
    secret: bytes = b"v" * 32,
    capabilities: tuple[AdmissionVerifierCapability, ...] | None = None,
) -> TrustedAdmissionVerifier:
    return TrustedAdmissionVerifier(
        identity=_verifier(capabilities),
        key_id="local-test-key-v1",
        secret=secret,
    )


def _trusted_interface(
    *,
    interface_bytes: bytes = INTERFACE_BYTES,
) -> TrustedRuntimeInterface:
    return TrustedRuntimeInterface(
        declared_interface=INTERFACE_NAME,
        interface_artifact=_content_artifact("example-interface-contract", interface_bytes),
        runtime_interface=ExampleRuntimeInterface,
    )


def _trusted_abc_interface() -> TrustedRuntimeInterface:
    return TrustedRuntimeInterface(
        declared_interface=ABSTRACT_INTERFACE_NAME,
        interface_artifact=_content_artifact("abstract-interface-contract", INTERFACE_BYTES),
        runtime_interface=AbstractRuntimeInterface,
    )


def _bundle_for_interface(interface_name: str) -> BiologicalModelBundleContract:
    payload = _bundle().model_dump(mode="python")
    for port in payload["ports"]:
        if port["implementation"] is not None:
            port["implementation"]["interface"] = interface_name
    for operation in payload["operation_implementations"]:
        operation["implementation"]["interface"] = interface_name
    return BiologicalModelBundleContract.model_validate(payload)


def _trusted_jit_loader(loaded_object: object = GoodImplementation) -> TrustedJITLoader:
    def load_exact(_receipt, sealed_code: bytes) -> object:
        assert sealed_code == CODE_BYTES
        return loaded_object

    return TrustedJITLoader(
        verifier=_trusted_verifier(),
        load_exact=load_exact,
    )


def _jit_loader_with_verifier(
    verifier: TrustedAdmissionVerifier,
    loaded_object: object = GoodImplementation,
) -> TrustedJITLoader:
    return TrustedJITLoader(
        verifier=verifier,
        load_exact=lambda _receipt, _sealed_code: loaded_object,
    )


def _runtime_interfaces() -> dict[str, TrustedRuntimeInterface]:
    return {INTERFACE_NAME: _trusted_interface()}


def _evidence(name: str) -> ContentAddressedArtifact:
    return _content_artifact(name, f"audit-log:{name}".encode())


def _source_selection(
    bundle: BiologicalModelBundleContract,
    *,
    source: SourceArtifact | None = None,
    trusted_selector: TrustedAdmissionVerifier | None = None,
):
    return issue_execution_source_selection_receipt(
        selection_id="benchmark-execution-source-selection",
        bundle=bundle,
        workflow_resolution_artifacts=(_content_artifact("workflow-resolution", WORKFLOW_BYTES),),
        sources=(source or _source_artifact(),),
        trusted_selector=trusted_selector or _trusted_verifier(),
        issued_at=NOW,
    )


def _declaration_bytes(bundle: BiologicalModelBundleContract) -> dict[str, bytes]:
    implementation = _implementation()
    return {
        bundle.query.artifact.sha256: b"query",
        bundle.benchmark.artifact.sha256: b"benchmark",
        bundle.support_envelope.artifact.sha256: b"envelope",
        implementation.code_artifact.sha256: CODE_BYTES,
        _trusted_interface().interface_artifact.sha256: INTERFACE_BYTES,
        _content_artifact("workflow-resolution", WORKFLOW_BYTES).sha256: WORKFLOW_BYTES,
        _source_artifact().sha256: SOURCE_BYTES,
    }


def _artifact_receipts(bundle: BiologicalModelBundleContract, selection) -> tuple:
    content_by_sha = _declaration_bytes(bundle)
    receipts = []
    for index, reference in enumerate(
        consumed_artifacts_for_bundle(
            bundle,
            execution_source_selection=selection,
            runtime_interfaces=_runtime_interfaces(),
        )
    ):
        declaration = (
            reference.content_addressed_artifact
            if reference.content_addressed_artifact is not None
            else reference.dataset_source_artifact
        )
        assert declaration is not None
        receipts.append(
            issue_artifact_resolution_receipt(
                receipt_id=f"artifact-receipt-{index:02d}",
                artifact=declaration,
                observed_content=BytesIO(content_by_sha[reference.sha256]),
                trusted_verifier=_trusted_verifier(),
                issued_at=NOW,
                evidence_artifacts=(_evidence(f"artifact-audit-{index:02d}"),),
            )
        )
    return tuple(receipts)


def _isolated_observation(
    requirement: ImplementationReceiptRequirement,
    *,
    observation_id: str,
    loaded_identity: LoadedObjectIdentity | None = None,
    trusted_loader: TrustedAdmissionVerifier | None = None,
    verification_method: InterfaceVerificationMethod = (
        InterfaceVerificationMethod.INSPECTED_SIGNATURE_SET
    ),
):
    identity = loaded_identity or LoadedObjectIdentity(
        entrypoint=ENTRYPOINT,
        module="cellstate_receipt_test_runtime",
        qualname="GoodImplementation",
        object_kind=LoadedObjectKind.CLASS,
        loaded_code_sha256=requirement.implementation.code_artifact.sha256,
    )
    conformance = InterfaceConformanceObservation(
        verification_method=verification_method,
        declared_interface=requirement.implementation.interface,
        runtime_interface_module=requirement.runtime_interface_module,
        runtime_interface_qualname=requirement.runtime_interface_qualname,
        loaded_object_fingerprint=identity.fingerprint,
        required_member_signatures=requirement.required_member_signatures,
        observed_matching_member_signatures=requirement.required_member_signatures,
        interface_contract_fingerprint=requirement.interface_contract_fingerprint,
        observed_contract_fingerprint=requirement.interface_contract_fingerprint,
    )
    return attest_isolated_loaded_interface_observation(
        observation_id=observation_id,
        requirement=requirement,
        loaded_object=identity,
        conformance=conformance,
        trusted_loader=trusted_loader or _trusted_verifier(),
        issued_at=NOW,
        isolation_evidence_artifact=_evidence(f"{observation_id}-isolation"),
    )


def _interface_receipts(bundle: BiologicalModelBundleContract) -> tuple:
    trusted_interface = _trusted_interface()
    requirements = implementation_requirements_for_bundle(
        bundle,
        runtime_interfaces=_runtime_interfaces(),
    )
    return tuple(
        issue_loaded_interface_receipt(
            receipt_id=f"interface-receipt-{index:02d}",
            requirement=requirement,
            isolated_observation=_isolated_observation(
                requirement,
                observation_id=f"isolated-observation-{index:02d}",
            ),
            trusted_runtime_interface=trusted_interface,
            trusted_loader=_trusted_verifier(),
            trusted_verifier=_trusted_verifier(),
            issued_at=NOW,
            evidence_artifacts=(_evidence(f"interface-audit-{index:02d}"),),
        )
        for index, requirement in enumerate(requirements)
    )


def _complete_report() -> tuple[
    BiologicalModelBundleContract,
    SourceArtifact,
    AdmissionReceiptBatchReport,
]:
    bundle = _bundle()
    source = _source_artifact()
    selection = _source_selection(bundle, source=source)
    report = build_admission_receipt_batch_report(
        batch_id="complete-receipt-batch",
        bundle=bundle,
        artifact_receipts=_artifact_receipts(bundle, selection),
        interface_receipts=_interface_receipts(bundle),
        runtime_interfaces=_runtime_interfaces(),
        execution_source_selection=selection,
        trusted_execution_source_selector=_trusted_verifier(),
        issued_at=NOW,
        evidence_artifacts=(_evidence("batch-audit"),),
    )
    return bundle, source, report


def test_artifact_receipt_streams_bytes_and_preserves_source_identity() -> None:
    source = _source_artifact()
    receipt = issue_artifact_resolution_receipt(
        receipt_id="source-byte-receipt",
        artifact=source,
        observed_content=(chunk for chunk in (b"real-", b"h5ad-", b"bytes")),
        trusted_verifier=_trusted_verifier(),
        issued_at=NOW,
        evidence_artifacts=(_evidence("source-byte-audit"),),
    )

    assert receipt.artifact.artifact_kind is AdmissionArtifactKind.DATASET_SOURCE_ARTIFACT
    assert receipt.artifact.dataset_source_artifact == source
    assert receipt.artifact.dataset_source_artifact.accession == "GSE139944"
    assert receipt.artifact.dataset_source_artifact.release == "2020-04-30"
    assert receipt.observation.observed_sha256 == source.sha256
    assert receipt.observation.observed_byte_count == source.byte_count

    with pytest.raises(ValidationError, match="SHA-256 must exactly match"):
        issue_artifact_resolution_receipt(
            receipt_id="wrong-source-byte-receipt",
            artifact=source,
            observed_content=BytesIO(b"wrong-h5ad-bytes"),
            trusted_verifier=_trusted_verifier(),
            issued_at=NOW,
            evidence_artifacts=(_evidence("wrong-source-byte-audit"),),
        )


def test_execution_sources_require_authenticated_typed_workflow_selection() -> None:
    bundle = _bundle()
    selection = _source_selection(bundle)
    assert selection.sources == (_source_artifact(),)
    assert selection.workflow_resolution_artifacts[0].artifact_id == "workflow-resolution"
    assert (
        require_valid_execution_source_selection(
            selection,
            trusted_selector=_trusted_verifier(),
        )
        == selection
    )

    with pytest.raises(ValueError, match="authentication failed"):
        require_valid_execution_source_selection(
            selection,
            trusted_selector=_trusted_verifier(secret=b"x" * 32),
        )
    with pytest.raises(ValidationError, match="at least 1 item"):
        issue_execution_source_selection_receipt(
            selection_id="empty-selection",
            bundle=bundle,
            workflow_resolution_artifacts=(
                _content_artifact("workflow-resolution", WORKFLOW_BYTES),
            ),
            sources=(),
            trusted_selector=_trusted_verifier(),
            issued_at=NOW,
        )


def test_generic_attestation_is_authenticated_and_capability_scoped() -> None:
    semantic_verifier = _trusted_verifier()
    payload = {"result_id": "semantic-result", "criterion_count": 12}
    attestation = attest_canonical_payload(
        payload,
        trusted_verifier=semantic_verifier,
        required_capability=AdmissionVerifierCapability.VALIDATION_RESULT_SEMANTICS,
    )
    require_valid_canonical_attestation(
        payload,
        attestation,
        verifier_identity=semantic_verifier.identity,
        trusted_verifier=semantic_verifier,
        required_capability=AdmissionVerifierCapability.VALIDATION_RESULT_SEMANTICS,
    )
    artifact_only = _trusted_verifier(
        capabilities=(AdmissionVerifierCapability.ARTIFACT_BYTE_RESOLUTION,)
    )
    with pytest.raises(ValueError, match="lacks required 'validation_result_semantics'"):
        attest_canonical_payload(
            payload,
            trusted_verifier=artifact_only,
            required_capability=AdmissionVerifierCapability.VALIDATION_RESULT_SEMANTICS,
        )


def test_interface_requirement_binds_registry_artifact_and_complete_signature_set() -> None:
    bundle = _bundle()
    requirements = implementation_requirements_for_bundle(
        bundle,
        runtime_interfaces=_runtime_interfaces(),
    )

    assert len(requirements) == 2
    assert all(
        requirement.interface_artifact == _trusted_interface().interface_artifact
        for requirement in requirements
    )
    assert all(
        any(
            signature.startswith("descriptor|property|")
            for signature in requirement.required_member_signatures
        )
        for requirement in requirements
    )
    assert all(
        any(
            signature.startswith("run|instance_method|")
            for signature in requirement.required_member_signatures
        )
        for requirement in requirements
    )

    first = requirements[0]
    trivial_members = (first.required_member_signatures[0],)
    forged_payload = first.model_dump(mode="python")
    forged_payload["required_member_signatures"] = trivial_members
    forged_payload["interface_contract_fingerprint"] = canonical_fingerprint(
        {
            "interface": first.implementation.interface,
            "runtime_interface_module": first.runtime_interface_module,
            "runtime_interface_qualname": first.runtime_interface_qualname,
            "member_signatures": trivial_members,
        }
    )
    forged_payload["trusted_interface_fingerprint"] = canonical_fingerprint(
        {
            "declared_interface": first.implementation.interface,
            "interface_artifact": first.interface_artifact.model_dump(mode="json"),
            "runtime_interface_module": first.runtime_interface_module,
            "runtime_interface_qualname": first.runtime_interface_qualname,
            "required_member_signatures": trivial_members,
        }
    )
    forged = ImplementationReceiptRequirement.model_validate(forged_payload)
    with pytest.raises(ValueError, match="stale or substituted"):
        issue_loaded_interface_receipt(
            receipt_id="forged-trivial-interface-receipt",
            requirement=forged,
            isolated_observation=_isolated_observation(
                forged,
                observation_id="forged-trivial-observation",
            ),
            trusted_runtime_interface=_trusted_interface(),
            trusted_loader=_trusted_verifier(),
            trusted_verifier=_trusted_verifier(),
            issued_at=NOW,
            evidence_artifacts=(_evidence("forged-interface-audit"),),
        )


def test_interface_receipt_requires_authenticated_isolated_exact_loaded_identity() -> None:
    requirement = implementation_requirements_for_bundle(
        _bundle(),
        runtime_interfaces=_runtime_interfaces(),
    )[0]
    observation = _isolated_observation(
        requirement,
        observation_id="valid-isolated-observation",
    )
    receipt = issue_loaded_interface_receipt(
        receipt_id="loaded-interface-receipt",
        requirement=requirement,
        isolated_observation=observation,
        trusted_runtime_interface=_trusted_interface(),
        trusted_loader=_trusted_verifier(),
        trusted_verifier=_trusted_verifier(),
        issued_at=NOW,
        evidence_artifacts=(_evidence("loaded-interface-audit"),),
    )

    assert receipt.loaded_object.object_kind is LoadedObjectKind.CLASS
    assert receipt.loaded_object.module == "cellstate_receipt_test_runtime"
    assert receipt.loaded_object.qualname == "GoodImplementation"
    assert receipt.conformance.required_member_signatures == requirement.required_member_signatures
    assert (
        receipt.execution_recheck_requirement == "reload_exact_bytes_and_reverify_before_execution"
    )

    wrong_identity = receipt.loaded_object.model_copy(update={"qualname": "OtherImplementation"})
    wrong_observation = _isolated_observation(
        requirement,
        observation_id="wrong-entrypoint-observation",
        loaded_identity=wrong_identity,
    )
    with pytest.raises(ValueError, match="declared entrypoint"):
        issue_loaded_interface_receipt(
            receipt_id="wrong-entrypoint-interface-receipt",
            requirement=requirement,
            isolated_observation=wrong_observation,
            trusted_runtime_interface=_trusted_interface(),
            trusted_loader=_trusted_verifier(),
            trusted_verifier=_trusted_verifier(),
            issued_at=NOW,
            evidence_artifacts=(_evidence("wrong-entrypoint-audit"),),
        )

    wrong_loader = _trusted_verifier(secret=b"z" * 32)
    untrusted_observation = _isolated_observation(
        requirement,
        observation_id="untrusted-loader-observation",
        trusted_loader=wrong_loader,
    )
    with pytest.raises(ValueError, match="authentication failed"):
        issue_loaded_interface_receipt(
            receipt_id="untrusted-loader-interface-receipt",
            requirement=requirement,
            isolated_observation=untrusted_observation,
            trusted_runtime_interface=_trusted_interface(),
            trusted_loader=_trusted_verifier(),
            trusted_verifier=_trusted_verifier(),
            issued_at=NOW,
            evidence_artifacts=(_evidence("untrusted-loader-audit"),),
        )


def test_jit_handle_rederives_real_object_bytes_and_descriptor_signatures() -> None:
    receipt = _interface_receipts(_bundle())[0]
    handle = reverify_jit_loaded_interface(
        receipt,
        resolved_code_content=BytesIO(CODE_BYTES),
        trusted_runtime_interface=_trusted_interface(),
        trusted_jit_loader=_trusted_jit_loader(),
    )
    assert handle.loaded_object is GoodImplementation
    assert handle.loaded_object_identity == receipt.loaded_object
    assert handle.observed_code_sha256 == receipt.requirement.implementation.code_artifact.sha256
    assert handle.target_kind is receipt.requirement.target_kind
    assert handle.target_id == receipt.requirement.target_id

    with pytest.raises(ValueError, match="JIT code bytes do not match"):
        reverify_jit_loaded_interface(
            receipt,
            resolved_code_content=BytesIO(b"different code"),
            trusted_runtime_interface=_trusted_interface(),
            trusted_jit_loader=_trusted_jit_loader(),
        )
    with pytest.raises(ValueError, match="signatures differ"):
        reverify_jit_loaded_interface(
            receipt,
            resolved_code_content=BytesIO(CODE_BYTES),
            trusted_runtime_interface=_trusted_interface(),
            trusted_jit_loader=_trusted_jit_loader(StaticMethodSubstitution),
        )
    with pytest.raises(ValueError, match="unimplemented interface member"):
        reverify_jit_loaded_interface(
            receipt,
            resolved_code_content=BytesIO(CODE_BYTES),
            trusted_runtime_interface=_trusted_interface(),
            trusted_jit_loader=_trusted_jit_loader(InheritedProtocolStub),
        )
    wrong_key_loader = _jit_loader_with_verifier(
        TrustedAdmissionVerifier(
            identity=_trusted_verifier().identity,
            key_id="different-loader-key",
            secret=b"v" * 32,
        )
    )
    with pytest.raises(ValueError, match="JIT loader key differs"):
        reverify_jit_loaded_interface(
            receipt,
            resolved_code_content=BytesIO(CODE_BYTES),
            trusted_runtime_interface=_trusted_interface(),
            trusted_jit_loader=wrong_key_loader,
        )


def test_jit_caller_cannot_pair_admitted_bytes_with_an_independent_object() -> None:
    receipt = _interface_receipts(_bundle())[0]

    with pytest.raises(TypeError, match="unexpected keyword argument 'loaded_object'"):
        reverify_jit_loaded_interface(  # type: ignore[call-arg]
            receipt,
            loaded_object=StaticMethodSubstitution,
            resolved_code_content=BytesIO(CODE_BYTES),
            trusted_runtime_interface=_trusted_interface(),
            trusted_jit_loader=_trusted_jit_loader(),
        )


def test_jit_abc_conformance_requires_nominal_subclassing() -> None:
    bundle = _bundle_for_interface(ABSTRACT_INTERFACE_NAME)
    trusted_interface = _trusted_abc_interface()
    requirement = implementation_requirements_for_bundle(
        bundle,
        runtime_interfaces={ABSTRACT_INTERFACE_NAME: trusted_interface},
    )[0]
    receipt = issue_loaded_interface_receipt(
        receipt_id="abstract-interface-receipt",
        requirement=requirement,
        isolated_observation=_isolated_observation(
            requirement,
            observation_id="abstract-interface-observation",
            verification_method=InterfaceVerificationMethod.ABSTRACT_BASE_CLASS,
        ),
        trusted_runtime_interface=trusted_interface,
        trusted_loader=_trusted_verifier(),
        trusted_verifier=_trusted_verifier(),
        issued_at=NOW,
        evidence_artifacts=(_evidence("abstract-interface-audit"),),
    )

    handle = reverify_jit_loaded_interface(
        receipt,
        resolved_code_content=CODE_BYTES,
        trusted_runtime_interface=trusted_interface,
        trusted_jit_loader=_trusted_jit_loader(NominalABCImplementation),
    )
    assert handle.loaded_object is NominalABCImplementation

    with pytest.raises(ValueError, match="nominal runtime interface conformance"):
        reverify_jit_loaded_interface(
            receipt,
            resolved_code_content=CODE_BYTES,
            trusted_runtime_interface=trusted_interface,
            trusted_jit_loader=_trusted_jit_loader(SignatureOnlyABCImpostor),
        )


def test_specification_only_implementation_cannot_acquire_interface_requirement() -> None:
    executable_requirement = implementation_requirements_for_bundle(
        _bundle(),
        runtime_interfaces=_runtime_interfaces(),
    )[0]
    specification_only = executable_requirement.implementation.model_copy(
        update={"kind": PortImplementationKind.SPECIFICATION_ONLY, "entrypoint": None}
    )
    payload = executable_requirement.model_dump(mode="python")
    payload["implementation"] = specification_only
    with pytest.raises(ValidationError, match="specification-only"):
        ImplementationReceiptRequirement.model_validate(payload)


def test_complete_batch_auto_covers_source_workflow_and_interface_artifacts_once() -> None:
    bundle, source, report = _complete_report()

    assert (
        require_exact_receipt_batch_coverage(
            bundle,
            report,
            trusted_verifiers=(_trusted_verifier(),),
            runtime_interfaces=_runtime_interfaces(),
        )
        == report
    )
    assert len(report.required_interfaces) == 2
    assert len(report.interface_receipts) == 2
    assert (
        sum(
            artifact.artifact_kind is AdmissionArtifactKind.DATASET_SOURCE_ARTIFACT
            for artifact in report.required_artifacts
        )
        == 1
    )
    assert report.artifact_receipt_for(source).artifact.dataset_source_artifact == source
    assert (
        report.artifact_receipt_for(
            _trusted_interface().interface_artifact
        ).artifact.content_addressed_artifact
        == _trusted_interface().interface_artifact
    )
    assert (
        report.artifact_receipt_for(
            report.execution_source_selection.workflow_resolution_artifacts[0]
        ).artifact.content_addressed_artifact
        == report.execution_source_selection.workflow_resolution_artifacts[0]
    )
    round_tripped = AdmissionReceiptBatchReport.model_validate_json(report.model_dump_json())
    assert round_tripped == report


def test_batch_rejects_missing_extra_and_duplicate_receipts() -> None:
    bundle = _bundle()
    selection = _source_selection(bundle)
    artifacts = _artifact_receipts(bundle, selection)
    interfaces = _interface_receipts(bundle)
    common = {
        "batch_id": "coverage-failure-batch",
        "bundle": bundle,
        "runtime_interfaces": _runtime_interfaces(),
        "execution_source_selection": selection,
        "trusted_execution_source_selector": _trusted_verifier(),
        "issued_at": NOW,
        "evidence_artifacts": (_evidence("coverage-failure-batch-audit"),),
    }

    with pytest.raises(ValidationError, match="cover every required exact artifact once"):
        build_admission_receipt_batch_report(
            artifact_receipts=artifacts[:-1],
            interface_receipts=interfaces,
            **common,
        )
    duplicate_artifact_receipt = issue_artifact_resolution_receipt(
        receipt_id="zz-duplicate-artifact-receipt",
        artifact=bundle.query.artifact,
        observed_content=b"query",
        trusted_verifier=_trusted_verifier(),
        issued_at=NOW,
        evidence_artifacts=(_evidence("duplicate-artifact-audit"),),
    )
    with pytest.raises(ValidationError, match="only one resolution receipt"):
        build_admission_receipt_batch_report(
            artifact_receipts=(*artifacts, duplicate_artifact_receipt),
            interface_receipts=interfaces,
            **common,
        )
    duplicate_interface_receipt = issue_loaded_interface_receipt(
        receipt_id="zz-duplicate-interface-receipt",
        requirement=interfaces[0].requirement,
        isolated_observation=_isolated_observation(
            interfaces[0].requirement,
            observation_id="duplicate-interface-observation",
        ),
        trusted_runtime_interface=_trusted_interface(),
        trusted_loader=_trusted_verifier(),
        trusted_verifier=_trusted_verifier(),
        issued_at=NOW,
        evidence_artifacts=(_evidence("duplicate-interface-audit"),),
    )
    with pytest.raises(ValidationError, match="implementation target may have only one"):
        build_admission_receipt_batch_report(
            artifact_receipts=artifacts,
            interface_receipts=(*interfaces, duplicate_interface_receipt),
            **common,
        )
    with pytest.raises(ValidationError, match="cover every exact implementation target once"):
        build_admission_receipt_batch_report(
            artifact_receipts=artifacts,
            interface_receipts=interfaces[:-1],
            **common,
        )

    extra_source = _source_artifact(b"different-source-bytes").model_copy(
        update={"source_id": "unselected-review-only-source"}
    )
    extra_receipt = issue_artifact_resolution_receipt(
        receipt_id="zz-unselected-source-receipt",
        artifact=extra_source,
        observed_content=b"different-source-bytes",
        trusted_verifier=_trusted_verifier(),
        issued_at=NOW,
        evidence_artifacts=(_evidence("unselected-source-audit"),),
    )
    with pytest.raises(ValidationError, match="cover every required exact artifact once"):
        build_admission_receipt_batch_report(
            artifact_receipts=(*artifacts, extra_receipt),
            interface_receipts=interfaces,
            **common,
        )


def _forge_publicly_rehashed_receipt(
    receipt: ArtifactResolutionReceipt,
) -> ArtifactResolutionReceipt:
    payload = receipt.model_dump(mode="python")
    payload["receipt_id"] = f"{receipt.receipt_id}-forged"
    attested_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"attestation", "receipt_fingerprint"}
    }
    payload["attestation"] = ReceiptAttestation(
        key_id=receipt.attestation.key_id,
        attested_payload_fingerprint=canonical_fingerprint(attested_payload),
        authentication_tag="0" * 64,
    )
    receipt_payload = {key: value for key, value in payload.items() if key != "receipt_fingerprint"}
    payload["receipt_fingerprint"] = canonical_fingerprint(receipt_payload)
    return ArtifactResolutionReceipt.model_validate(payload)


def test_rebinding_revalidates_model_copy_and_rejects_publicly_rehashed_forgery() -> None:
    bundle, _, report = _complete_report()
    model_copy_tamper = report.model_copy(update={"report_fingerprint": "0" * 64})
    with pytest.raises(ValidationError, match="batch fingerprint"):
        require_exact_receipt_batch_coverage(
            bundle,
            model_copy_tamper,
            trusted_verifiers=(_trusted_verifier(),),
            runtime_interfaces=_runtime_interfaces(),
        )

    forged = _forge_publicly_rehashed_receipt(report.artifact_receipts[0])
    forged_report = build_admission_receipt_batch_report(
        batch_id="forged-receipt-batch",
        bundle=bundle,
        artifact_receipts=(forged, *report.artifact_receipts[1:]),
        interface_receipts=report.interface_receipts,
        runtime_interfaces=_runtime_interfaces(),
        execution_source_selection=report.execution_source_selection,
        trusted_execution_source_selector=_trusted_verifier(),
        issued_at=NOW,
        evidence_artifacts=(_evidence("forged-batch-audit"),),
    )
    with pytest.raises(ValueError, match="attestation authentication failed"):
        require_exact_receipt_batch_coverage(
            bundle,
            forged_report,
            trusted_verifiers=(_trusted_verifier(),),
            runtime_interfaces=_runtime_interfaces(),
        )


def test_rebinding_rejects_stale_scope_and_substituted_runtime_registry() -> None:
    bundle, _, report = _complete_report()
    changed_bundle = _bundle(b"changed-implementation-code")
    with pytest.raises(ValueError, match="stale implementation scope"):
        require_exact_receipt_batch_coverage(
            changed_bundle,
            report,
            trusted_verifiers=(_trusted_verifier(),),
            runtime_interfaces=_runtime_interfaces(),
        )

    substituted_registry = {
        INTERFACE_NAME: _trusted_interface(interface_bytes=b"substituted interface contract")
    }
    with pytest.raises(ValueError, match=r"exact artifacts|exact interfaces"):
        require_exact_receipt_batch_coverage(
            bundle,
            report,
            trusted_verifiers=(_trusted_verifier(),),
            runtime_interfaces=substituted_registry,
        )
