"""Adversarial tests for authenticated validation-result semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_item9_model_bundle_contract import complete_looking_runtime_contracts

from cellstate.backends.admission import (
    AdmissionReceiptBatchReport,
    AdmissionVerifierCapability,
    AdmissionVerifierIdentity,
    TrustedAdmissionVerifier,
    admission_artifact_reference,
    artifact_coverage_fingerprint,
    interface_coverage_fingerprint,
    issue_artifact_resolution_receipt,
    issue_execution_source_selection_receipt,
)
from cellstate.backends.contracts import (
    BiologicalModelBundleContract,
    BiologicalStagePort,
    BiologicalSupportEnvelope,
    BundleContractReference,
    ModelOperation,
    ValidationEvidenceBinding,
)
from cellstate.backends.validation import (
    REQUIRED_VALIDATION_CRITERIA,
    VALIDATION_RESULT_MEDIA_TYPE,
    ValidationCriterionResult,
    ValidationResultManifest,
    ValidationResultReceiptBatch,
    ValidationResultVerificationReceipt,
    ValidationSemanticCriterion,
    build_validation_result_receipt_batch,
    issue_validation_result_verification_receipt,
    require_exact_validation_result_coverage,
    required_validation_criteria,
)
from cellstate.data import BenchmarkArtifact, ContentAddressedArtifact, SourceArtifact, SourceKind
from cellstate.domain import CriterionOutcome, StateQuery
from cellstate.domain.common import canonical_fingerprint, canonical_json_bytes

ROOT = Path(__file__).resolve().parents[1]
QUERY_PATH = ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/state-query.json"
BENCHMARK_PATH = ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/benchmark-artifact.json"
NOW = datetime(2026, 8, 10, 18, 0, tzinfo=UTC)

DETAILS_BYTES = b'{"authoritative_cases":true,"frozen_protocol":true}'
WORKFLOW_BYTES = b"typed validation assessment and permission resolution v1"
SOURCE_BYTES = b"exact checked-in benchmark source bytes"


@pytest.fixture(scope="module")
def query() -> StateQuery:
    """Load the checked-in sciPlex query rather than relying on another test's fixture."""

    return StateQuery.model_validate_json(QUERY_PATH.read_bytes())


@pytest.fixture(scope="module")
def benchmark() -> BenchmarkArtifact:
    """Load the real checked-in sciPlex benchmark artifact."""

    return BenchmarkArtifact.model_validate_json(BENCHMARK_PATH.read_bytes())


def _artifact(
    artifact_id: str,
    content: bytes,
    *,
    media_type: str = "application/octet-stream",
) -> ContentAddressedArtifact:
    return ContentAddressedArtifact(
        artifact_id=artifact_id,
        uri=f"https://example.invalid/validation/{artifact_id}",
        sha256=sha256(content).hexdigest(),
        byte_count=len(content),
        media_type=media_type,
    )


def _source() -> SourceArtifact:
    return SourceArtifact(
        source_id="sciplex3-validation-source",
        kind=SourceKind.PROCESSED,
        uri="https://example.invalid/validation/sciplex3.h5ad",
        sha256=sha256(SOURCE_BYTES).hexdigest(),
        media_type="application/x-hdf5",
        accession="GSE139944",
        release="2020-04-30",
        parent_study_accession="GSE139944",
        parent_study_release="2020-04-30",
        byte_count=len(SOURCE_BYTES),
        retrieved_at=NOW,
    )


def _identity(
    name: str,
    capability: AdmissionVerifierCapability,
    *,
    version: str = "0.1.0",
) -> AdmissionVerifierIdentity:
    code = f"{name}:{version}:{capability.value}".encode()
    return AdmissionVerifierIdentity(
        verifier_id=f"cellstate.tests.{name}",
        verifier_version=version,
        code_artifact=_artifact(f"{name}-verifier-code-{version}", code),
        entrypoint=f"cellstate.tests.validation:{name}",
        runtime="cpython-3.11",
        capabilities=(capability,),
    )


def _trusted(
    name: str,
    capability: AdmissionVerifierCapability,
    *,
    secret: bytes,
    key_id: str,
    version: str = "0.1.0",
) -> TrustedAdmissionVerifier:
    return TrustedAdmissionVerifier(
        identity=_identity(name, capability, version=version),
        key_id=key_id,
        secret=secret,
    )


def _contract_reference(
    contract_id: str,
    contract_version: str,
    payload: ValidationEvidenceBinding,
) -> BundleContractReference:
    content = canonical_json_bytes(payload.model_dump(mode="json"))
    return BundleContractReference(
        contract_id=contract_id,
        contract_version=contract_version,
        artifact=ContentAddressedArtifact(
            artifact_id=f"{contract_id}-canonical-json",
            uri=f"https://example.invalid/contracts/{contract_id}/{contract_version}.json",
            sha256=payload.fingerprint,
            byte_count=len(content),
            media_type="application/json",
        ),
    )


def _criteria(
    evidence: ValidationEvidenceBinding,
    outcome: CriterionOutcome,
) -> tuple[ValidationCriterionResult, ...]:
    required = required_validation_criteria(
        evidence_kind=evidence.evidence_kind,
        covered_ports=evidence.covered_ports,
        covered_operations=evidence.covered_operations,
    )
    return tuple(
        ValidationCriterionResult(
            criterion=criterion,
            outcome=outcome,
            evidence_artifact_ids=("semantic-result-details",)
            if outcome in {CriterionOutcome.PASSED, CriterionOutcome.FAILED}
            else (),
            reasons=("Frozen validation criterion did not pass.",)
            if outcome is not CriterionOutcome.PASSED
            else (),
        )
        for criterion in sorted(required, key=lambda item: item.value)
    )


@dataclass(frozen=True, slots=True)
class PreparedValidation:
    envelope: BiologicalSupportEnvelope
    bundle: BiologicalModelBundleContract
    evidence: ValidationEvidenceBinding
    manifest: ValidationResultManifest
    manifest_bytes: bytes
    details_artifact: ContentAddressedArtifact
    manifest_artifact: ContentAddressedArtifact


@dataclass(frozen=True, slots=True)
class ValidationScenario:
    prepared: PreparedValidation
    byte_verifier: TrustedAdmissionVerifier
    selector: TrustedAdmissionVerifier
    evaluator: TrustedAdmissionVerifier
    authoritative_batch: AdmissionReceiptBatchReport
    semantic_receipt: ValidationResultVerificationReceipt
    semantic_batch: ValidationResultReceiptBatch

    @property
    def trust_roots(self) -> tuple[TrustedAdmissionVerifier, ...]:
        return (self.byte_verifier, self.selector, self.evaluator)


def _prepare_validation(
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    *,
    outcome: CriterionOutcome,
    canonical_encoding: bool = True,
) -> PreparedValidation:
    envelope, _, original_evidence, original_bundle = complete_looking_runtime_contracts(
        query,
        benchmark,
    )
    details_artifact = _artifact(
        "semantic-result-details",
        DETAILS_BYTES,
        media_type="application/json",
    )
    scope_payload = original_evidence.model_dump(mode="python")
    scope_payload["evidence_artifacts"] = (details_artifact,)
    scope_evidence = ValidationEvidenceBinding.model_validate(scope_payload)
    manifest = ValidationResultManifest(
        result_id="synthetic-validation-result",
        result_version="0.1.0",
        validation_scope_fingerprint=scope_evidence.validation_scope_fingerprint,
        evidence_id=scope_evidence.evidence_id,
        evidence_version=scope_evidence.evidence_version,
        evidence_kind=scope_evidence.evidence_kind,
        query_fingerprint=scope_evidence.query_fingerprint,
        benchmark_fingerprint=scope_evidence.benchmark_fingerprint,
        support_envelope_fingerprint=scope_evidence.support_envelope_fingerprint,
        training_run_fingerprint=scope_evidence.training_run_fingerprint,
        model_artifact_fingerprint=scope_evidence.model_artifact_fingerprint,
        implementation_scope_fingerprint=scope_evidence.implementation_scope_fingerprint,
        partition_ids=scope_evidence.partition_ids,
        evaluation_case_ids=scope_evidence.evaluation_case_ids,
        covered_ports=scope_evidence.covered_ports,
        covered_operations=scope_evidence.covered_operations,
        result_manifest_artifact_id="semantic-result-manifest",
        supporting_artifacts=(details_artifact,),
        criteria=_criteria(scope_evidence, outcome),
        generated_at=NOW,
    )
    canonical_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
    manifest_bytes = canonical_bytes if canonical_encoding else b" " + canonical_bytes
    manifest_artifact = _artifact(
        "semantic-result-manifest",
        manifest_bytes,
        media_type=VALIDATION_RESULT_MEDIA_TYPE,
    )
    evidence_payload = scope_evidence.model_dump(mode="python")
    evidence_payload["evidence_artifacts"] = (details_artifact, manifest_artifact)
    evidence = ValidationEvidenceBinding.model_validate(evidence_payload)
    assert evidence.validation_scope_fingerprint == manifest.validation_scope_fingerprint

    bundle_payload = original_bundle.model_dump(mode="python")
    bundle_payload["validation_evidence"] = (
        _contract_reference(evidence.evidence_id, evidence.evidence_version, evidence),
    )
    bundle = BiologicalModelBundleContract.model_validate(bundle_payload)
    return PreparedValidation(
        envelope=envelope,
        bundle=bundle,
        evidence=evidence,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        details_artifact=details_artifact,
        manifest_artifact=manifest_artifact,
    )


def _authoritative_batch(
    prepared: PreparedValidation,
    *,
    byte_verifier: TrustedAdmissionVerifier,
    selector: TrustedAdmissionVerifier,
) -> AdmissionReceiptBatchReport:
    workflow_artifact = _artifact("validation-workflow-resolution", WORKFLOW_BYTES)
    source = _source()
    selection = issue_execution_source_selection_receipt(
        selection_id="validation-execution-source-selection",
        bundle=prepared.bundle,
        workflow_resolution_artifacts=(workflow_artifact,),
        sources=(source,),
        trusted_selector=selector,
        issued_at=NOW,
    )
    declarations = (
        prepared.details_artifact,
        prepared.manifest_artifact,
        workflow_artifact,
        source,
    )
    content_by_id = {
        prepared.details_artifact.artifact_id: DETAILS_BYTES,
        prepared.manifest_artifact.artifact_id: prepared.manifest_bytes,
        workflow_artifact.artifact_id: WORKFLOW_BYTES,
        source.source_id: SOURCE_BYTES,
    }
    references = tuple(
        sorted(
            (admission_artifact_reference(item) for item in declarations),
            key=lambda item: item.target_key,
        )
    )
    artifact_receipts = tuple(
        issue_artifact_resolution_receipt(
            receipt_id=f"authoritative-artifact-{index:02d}",
            artifact=(
                reference.content_addressed_artifact
                if reference.content_addressed_artifact is not None
                else reference.dataset_source_artifact
            ),
            observed_content=content_by_id[reference.reference_id],
            trusted_verifier=byte_verifier,
            issued_at=NOW,
            evidence_artifacts=(_artifact(f"authoritative-artifact-audit-{index:02d}", b"audit"),),
        )
        for index, reference in enumerate(references)
    )
    payload = {
        "schema_version": "0.1-experimental",
        "report_kind": "complete_admission_receipt_batch",
        "batch_id": "authoritative-validation-byte-batch",
        "bundle_fingerprint": prepared.bundle.fingerprint,
        "implementation_scope_fingerprint": prepared.bundle.implementation_scope_fingerprint,
        "execution_source_selection": selection,
        "required_artifacts": references,
        "artifact_receipts": artifact_receipts,
        "required_interfaces": (),
        "interface_receipts": (),
        "artifact_coverage_fingerprint": artifact_coverage_fingerprint(references),
        "interface_coverage_fingerprint": interface_coverage_fingerprint(()),
        "issued_at": NOW,
        "evidence_artifacts": (_artifact("authoritative-validation-batch-audit", b"audit"),),
    }
    return AdmissionReceiptBatchReport.model_validate(
        {**payload, "report_fingerprint": canonical_fingerprint(payload)}
    )


def _scenario(
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    *,
    outcome: CriterionOutcome,
) -> ValidationScenario:
    prepared = _prepare_validation(query, benchmark, outcome=outcome)
    byte_verifier = _trusted(
        "byte-verifier",
        AdmissionVerifierCapability.ARTIFACT_BYTE_RESOLUTION,
        secret=b"b" * 32,
        key_id="byte-key-v1",
    )
    selector = _trusted(
        "source-selector",
        AdmissionVerifierCapability.EXECUTION_SOURCE_SELECTION,
        secret=b"s" * 32,
        key_id="selector-key-v1",
    )
    evaluator = _trusted(
        "semantic-evaluator",
        AdmissionVerifierCapability.VALIDATION_RESULT_SEMANTICS,
        secret=b"e" * 32,
        key_id="semantic-key-v1",
    )
    authoritative_batch = _authoritative_batch(
        prepared,
        byte_verifier=byte_verifier,
        selector=selector,
    )
    semantic_artifact_receipts = tuple(
        authoritative_batch.artifact_receipt_for(artifact)
        for artifact in prepared.evidence.evidence_artifacts
    )
    semantic_receipt = issue_validation_result_verification_receipt(
        receipt_id="semantic-validation-receipt",
        bundle=prepared.bundle,
        evidence=prepared.evidence,
        result_manifest_artifact_id=prepared.manifest_artifact.artifact_id,
        result_manifest_bytes=prepared.manifest_bytes,
        artifact_receipts=semantic_artifact_receipts,
        trusted_evaluator=evaluator,
        issued_at=NOW,
        isolation_evidence_artifact=_artifact("semantic-evaluator-isolation", b"isolation"),
    )
    semantic_batch = build_validation_result_receipt_batch(
        batch_id="semantic-validation-batch",
        bundle=prepared.bundle,
        validation_evidence=(prepared.evidence,),
        receipts=(semantic_receipt,),
        issued_at=NOW,
    )
    return ValidationScenario(
        prepared=prepared,
        byte_verifier=byte_verifier,
        selector=selector,
        evaluator=evaluator,
        authoritative_batch=authoritative_batch,
        semantic_receipt=semantic_receipt,
        semantic_batch=semantic_batch,
    )


@pytest.fixture(scope="module")
def passed_scenario(query: StateQuery, benchmark: BenchmarkArtifact) -> ValidationScenario:
    return _scenario(query, benchmark, outcome=CriterionOutcome.PASSED)


@pytest.fixture(scope="module")
def failed_scenario(query: StateQuery, benchmark: BenchmarkArtifact) -> ValidationScenario:
    return _scenario(query, benchmark, outcome=CriterionOutcome.FAILED)


def _verify(scenario: ValidationScenario) -> ValidationResultReceiptBatch:
    return require_exact_validation_result_coverage(
        scenario.prepared.bundle,
        scenario.prepared.envelope,
        (scenario.prepared.evidence,),
        scenario.semantic_batch,
        trusted_verifiers=scenario.trust_roots,
        authoritative_receipt_batch=scenario.authoritative_batch,
    )


@pytest.mark.parametrize(
    ("fixture_name", "expected_passed"),
    (("passed_scenario", True), ("failed_scenario", False)),
)
def test_authenticated_results_are_verified_independently_of_pass_fail_status(
    fixture_name: str,
    expected_passed: bool,
    request: pytest.FixtureRequest,
) -> None:
    scenario: ValidationScenario = request.getfixturevalue(fixture_name)

    assert scenario.semantic_receipt.passed is expected_passed
    assert scenario.semantic_batch.all_results_passed is expected_passed
    assert _verify(scenario) == scenario.semantic_batch


def test_semantic_receipts_require_capability_scoped_hmac_authority(
    passed_scenario: ValidationScenario,
) -> None:
    prepared = passed_scenario.prepared
    artifact_only = _trusted(
        "artifact-only-evaluator",
        AdmissionVerifierCapability.ARTIFACT_BYTE_RESOLUTION,
        secret=b"a" * 32,
        key_id="artifact-only-key",
    )
    with pytest.raises(ValueError, match="lacks required 'validation_result_semantics'"):
        issue_validation_result_verification_receipt(
            receipt_id="wrong-capability-semantic-receipt",
            bundle=prepared.bundle,
            evidence=prepared.evidence,
            result_manifest_artifact_id=prepared.manifest_artifact.artifact_id,
            result_manifest_bytes=prepared.manifest_bytes,
            artifact_receipts=passed_scenario.semantic_receipt.artifact_receipts,
            trusted_evaluator=artifact_only,
            issued_at=NOW,
            isolation_evidence_artifact=_artifact("wrong-capability-isolation", b"isolation"),
        )


def test_result_manifest_bytes_must_be_exact_canonical_json(
    query: StateQuery,
    benchmark: BenchmarkArtifact,
) -> None:
    prepared = _prepare_validation(
        query,
        benchmark,
        outcome=CriterionOutcome.PASSED,
        canonical_encoding=False,
    )
    byte_verifier = _trusted(
        "noncanonical-byte-verifier",
        AdmissionVerifierCapability.ARTIFACT_BYTE_RESOLUTION,
        secret=b"n" * 32,
        key_id="noncanonical-byte-key",
    )
    selector = _trusted(
        "noncanonical-selector",
        AdmissionVerifierCapability.EXECUTION_SOURCE_SELECTION,
        secret=b"q" * 32,
        key_id="noncanonical-selector-key",
    )
    authoritative = _authoritative_batch(
        prepared,
        byte_verifier=byte_verifier,
        selector=selector,
    )
    evaluator = _trusted(
        "noncanonical-evaluator",
        AdmissionVerifierCapability.VALIDATION_RESULT_SEMANTICS,
        secret=b"m" * 32,
        key_id="noncanonical-semantic-key",
    )

    with pytest.raises(ValueError, match="canonical JSON encoding"):
        issue_validation_result_verification_receipt(
            receipt_id="noncanonical-semantic-receipt",
            bundle=prepared.bundle,
            evidence=prepared.evidence,
            result_manifest_artifact_id=prepared.manifest_artifact.artifact_id,
            result_manifest_bytes=prepared.manifest_bytes,
            artifact_receipts=tuple(
                authoritative.artifact_receipt_for(artifact)
                for artifact in prepared.evidence.evidence_artifacts
            ),
            trusted_evaluator=evaluator,
            issued_at=NOW,
            isolation_evidence_artifact=_artifact("noncanonical-isolation", b"isolation"),
        )


def test_criteria_are_closed_over_conditional_operation_calibration_and_ood_surfaces(
    passed_scenario: ValidationScenario,
) -> None:
    evidence = passed_scenario.prepared.evidence
    required = required_validation_criteria(
        evidence_kind=evidence.evidence_kind,
        covered_ports=evidence.covered_ports,
        covered_operations=evidence.covered_operations,
    )

    assert set(REQUIRED_VALIDATION_CRITERIA[evidence.evidence_kind]) < set(required)
    assert {
        ValidationSemanticCriterion.OPERATION_CONTRACT_CONFORMANCE,
        ValidationSemanticCriterion.CALIBRATION_BEHAVIOR,
        ValidationSemanticCriterion.OOD_ABSTENTION_BEHAVIOR,
        ValidationSemanticCriterion.SCIENTIFIC_SUPPORT_BOUNDS,
    }.issubset(required)
    assert BiologicalStagePort.UNCERTAINTY_CALIBRATOR in evidence.covered_ports
    assert BiologicalStagePort.STRICT_SUPPORT_OOD_GATE in evidence.covered_ports
    assert evidence.covered_operations == (ModelOperation.EVOLVE_CELL_STATE,)

    payload = passed_scenario.prepared.manifest.model_dump(mode="python")
    payload["criteria"] = tuple(
        item
        for item in payload["criteria"]
        if item["criterion"] is not ValidationSemanticCriterion.CALIBRATION_BEHAVIOR
    )
    with pytest.raises(ValidationError, match="exact criteria"):
        ValidationResultManifest.model_validate(payload)


def test_signed_failed_result_cannot_be_rewritten_to_passed_with_unkeyed_fingerprints(
    failed_scenario: ValidationScenario,
) -> None:
    prepared = failed_scenario.prepared
    forged_manifest = prepared.manifest.model_copy(
        update={"criteria": _criteria(prepared.evidence, CriterionOutcome.PASSED)}
    )
    unsigned_receipt = failed_scenario.semantic_receipt.model_copy(
        update={"result_manifest": forged_manifest}
    )
    forged_receipt = unsigned_receipt.model_copy(
        update={
            "receipt_fingerprint": canonical_fingerprint(
                unsigned_receipt.model_dump(mode="json", exclude={"receipt_fingerprint"})
            )
        }
    )
    unsigned_batch = failed_scenario.semantic_batch.model_copy(
        update={"receipts": (forged_receipt,)}
    )
    forged_batch = unsigned_batch.model_copy(
        update={
            "batch_fingerprint": canonical_fingerprint(
                unsigned_batch.model_dump(mode="json", exclude={"batch_fingerprint"})
            )
        }
    )
    assert forged_batch.all_results_passed

    with pytest.raises(ValueError, match="embedded validation result"):
        require_exact_validation_result_coverage(
            prepared.bundle,
            prepared.envelope,
            (prepared.evidence,),
            forged_batch,
            trusted_verifiers=failed_scenario.trust_roots,
            authoritative_receipt_batch=failed_scenario.authoritative_batch,
        )


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("secret", "authentication failed"),
        ("key", "untrusted evaluator"),
        ("identity", "untrusted evaluator"),
    ),
)
def test_semantic_receipt_rejects_wrong_external_trust_root(
    failure: str,
    message: str,
    passed_scenario: ValidationScenario,
) -> None:
    evaluator = passed_scenario.evaluator
    if failure == "secret":
        wrong = TrustedAdmissionVerifier(
            identity=evaluator.identity,
            key_id=evaluator.key_id,
            secret=b"x" * 32,
        )
    elif failure == "key":
        wrong = TrustedAdmissionVerifier(
            identity=evaluator.identity,
            key_id="wrong-semantic-key",
            secret=evaluator.secret,
        )
    else:
        wrong = _trusted(
            "other-semantic-evaluator",
            AdmissionVerifierCapability.VALIDATION_RESULT_SEMANTICS,
            secret=evaluator.secret,
            key_id=evaluator.key_id,
        )
    roots = (passed_scenario.byte_verifier, passed_scenario.selector, wrong)

    with pytest.raises(ValueError, match=message):
        require_exact_validation_result_coverage(
            passed_scenario.prepared.bundle,
            passed_scenario.prepared.envelope,
            (passed_scenario.prepared.evidence,),
            passed_scenario.semantic_batch,
            trusted_verifiers=roots,
            authoritative_receipt_batch=passed_scenario.authoritative_batch,
        )


def test_validation_scope_replay_is_rejected_before_semantic_attestation(
    passed_scenario: ValidationScenario,
) -> None:
    prepared = passed_scenario.prepared
    evidence_payload = prepared.evidence.model_dump(mode="python")
    evidence_payload["evaluation_case_ids"] = evidence_payload["evaluation_case_ids"][:-1]
    replay_scope = ValidationEvidenceBinding.model_validate(evidence_payload)

    with pytest.raises(ValueError, match="exact evidence binding"):
        issue_validation_result_verification_receipt(
            receipt_id="scope-replay-receipt",
            bundle=prepared.bundle,
            evidence=replay_scope,
            result_manifest_artifact_id=prepared.manifest_artifact.artifact_id,
            result_manifest_bytes=prepared.manifest_bytes,
            artifact_receipts=passed_scenario.semantic_receipt.artifact_receipts,
            trusted_evaluator=passed_scenario.evaluator,
            issued_at=NOW,
            isolation_evidence_artifact=_artifact("scope-replay-isolation", b"isolation"),
        )


def test_result_artifact_replay_from_another_signed_evaluation_is_rejected(
    passed_scenario: ValidationScenario,
    failed_scenario: ValidationScenario,
) -> None:
    prepared = passed_scenario.prepared
    with pytest.raises(ValidationError, match="every exact result artifact once"):
        issue_validation_result_verification_receipt(
            receipt_id="artifact-replay-receipt",
            bundle=prepared.bundle,
            evidence=prepared.evidence,
            result_manifest_artifact_id=prepared.manifest_artifact.artifact_id,
            result_manifest_bytes=prepared.manifest_bytes,
            artifact_receipts=failed_scenario.semantic_receipt.artifact_receipts,
            trusted_evaluator=passed_scenario.evaluator,
            issued_at=NOW,
            isolation_evidence_artifact=_artifact("artifact-replay-isolation", b"isolation"),
        )


def test_validation_batch_cannot_replay_across_bundle_identity(
    passed_scenario: ValidationScenario,
) -> None:
    payload = passed_scenario.prepared.bundle.model_dump(mode="python")
    payload["bundle_version"] = "0.1.1-drifted"
    drifted_bundle = BiologicalModelBundleContract.model_validate(payload)

    with pytest.raises(ValueError, match="different bundle"):
        require_exact_validation_result_coverage(
            drifted_bundle,
            passed_scenario.prepared.envelope,
            (passed_scenario.prepared.evidence,),
            passed_scenario.semantic_batch,
            trusted_verifiers=passed_scenario.trust_roots,
            authoritative_receipt_batch=passed_scenario.authoritative_batch,
        )


def test_semantic_receipt_must_reuse_the_authoritative_exact_byte_receipts(
    passed_scenario: ValidationScenario,
) -> None:
    prepared = passed_scenario.prepared
    content_by_id = {
        prepared.details_artifact.artifact_id: DETAILS_BYTES,
        prepared.manifest_artifact.artifact_id: prepared.manifest_bytes,
    }
    alternate_receipts = tuple(
        issue_artifact_resolution_receipt(
            receipt_id=f"alternate-semantic-artifact-{index:02d}",
            artifact=artifact,
            observed_content=content_by_id[artifact.artifact_id],
            trusted_verifier=passed_scenario.byte_verifier,
            issued_at=NOW,
            evidence_artifacts=(
                _artifact(f"alternate-semantic-audit-{index:02d}", b"alternate-audit"),
            ),
        )
        for index, artifact in enumerate(prepared.evidence.evidence_artifacts)
    )
    alternate_semantic_receipt = issue_validation_result_verification_receipt(
        receipt_id="alternate-byte-semantic-receipt",
        bundle=prepared.bundle,
        evidence=prepared.evidence,
        result_manifest_artifact_id=prepared.manifest_artifact.artifact_id,
        result_manifest_bytes=prepared.manifest_bytes,
        artifact_receipts=alternate_receipts,
        trusted_evaluator=passed_scenario.evaluator,
        issued_at=NOW,
        isolation_evidence_artifact=_artifact("alternate-byte-isolation", b"isolation"),
    )
    alternate_batch = build_validation_result_receipt_batch(
        batch_id="alternate-byte-semantic-batch",
        bundle=prepared.bundle,
        validation_evidence=(prepared.evidence,),
        receipts=(alternate_semantic_receipt,),
        issued_at=NOW,
    )

    with pytest.raises(ValueError, match="reuse the authoritative exact-byte receipt"):
        require_exact_validation_result_coverage(
            prepared.bundle,
            prepared.envelope,
            (prepared.evidence,),
            alternate_batch,
            trusted_verifiers=passed_scenario.trust_roots,
            authoritative_receipt_batch=passed_scenario.authoritative_batch,
        )


def test_validation_batch_fingerprint_cannot_be_rewritten(
    passed_scenario: ValidationScenario,
) -> None:
    payload = passed_scenario.semantic_batch.model_dump(mode="python")
    payload["batch_id"] = "rewritten-validation-batch"
    with pytest.raises(ValidationError, match="fingerprint does not match"):
        ValidationResultReceiptBatch.model_validate(payload)


@pytest.mark.parametrize("mutation", ("empty", "duplicate"))
def test_model_copy_cannot_bypass_closed_validation_receipt_coverage(
    mutation: str,
    passed_scenario: ValidationScenario,
) -> None:
    receipts = (
        ()
        if mutation == "empty"
        else (
            passed_scenario.semantic_receipt,
            passed_scenario.semantic_receipt,
        )
    )
    bypass = passed_scenario.semantic_batch.model_copy(update={"receipts": receipts})
    if mutation == "empty":
        assert bypass.all_results_passed

    with pytest.raises(ValidationError):
        require_exact_validation_result_coverage(
            passed_scenario.prepared.bundle,
            passed_scenario.prepared.envelope,
            (passed_scenario.prepared.evidence,),
            bypass,
            trusted_verifiers=passed_scenario.trust_roots,
            authoritative_receipt_batch=passed_scenario.authoritative_batch,
        )
