"""Compact failure matrix for the trusted Item 10 admission boundary."""

from __future__ import annotations

from datetime import timedelta

import pytest
import test_item10_admission_receipts as admission_cases
import test_item10_validation_results as validation_cases
from pydantic import ValidationError
from test_item9_model_bundle_contract import complete_looking_runtime_contracts

from cellstate.backends.admission import (
    AdmissionVerifierCapability,
    AdmissionVerifierIdentity,
    ImplementationReceiptTargetKind,
    TrustedAdmissionVerifier,
    TrustedJITLoader,
)
from cellstate.backends.contracts import (
    QUERY_PREREQUISITE_COMPILER_FINGERPRINT,
    QUERY_PREREQUISITE_COMPILER_ID,
    QUERY_PREREQUISITE_COMPILER_VERSION,
    BiologicalStagePort,
    ModelOperation,
    QueryDerivedPortPrerequisite,
    QueryDerivedPrerequisiteReport,
    QueryDerivedPrerequisiteTarget,
    QueryPrerequisiteReasonCode,
    QueryPrerequisiteTargetKind,
)
from cellstate.backends.validation import (
    AdmissionVerificationContext,
    ValidationCriterionResult,
    ValidationResultManifest,
    ValidationResultVerificationReceipt,
    admission_execution_artifacts,
    reverify_admission_jit_interfaces,
)
from cellstate.data import BenchmarkArtifact
from cellstate.domain import CriterionOutcome, StateQuery


@pytest.fixture(scope="module")
def validation_scenario():
    query = StateQuery.model_validate_json(validation_cases.QUERY_PATH.read_bytes())
    benchmark = BenchmarkArtifact.model_validate_json(validation_cases.BENCHMARK_PATH.read_bytes())
    return validation_cases._scenario(query, benchmark, outcome=CriterionOutcome.PASSED)


def _jit_report(bundle) -> QueryDerivedPrerequisiteReport:
    target = QueryDerivedPrerequisiteTarget(
        target_kind=QueryPrerequisiteTargetKind.RUNTIME_OPERATION,
        operation=ModelOperation.ESTIMATE_CELL_STATE,
        required_ports=(
            QueryDerivedPortPrerequisite(
                port=BiologicalStagePort.ACTION_CONTEXT_ENCODER,
                reasons=(QueryPrerequisiteReasonCode.OPERATION_CONTRACT_FLOOR,),
            ),
        ),
    )
    return QueryDerivedPrerequisiteReport(
        compiler_id=QUERY_PREREQUISITE_COMPILER_ID,
        compiler_version=QUERY_PREREQUISITE_COMPILER_VERSION,
        compiler_fingerprint=QUERY_PREREQUISITE_COMPILER_FINGERPRINT,
        query_fingerprint=bundle.query.artifact.sha256,
        support_envelope_id=bundle.support_envelope.contract_id,
        support_envelope_version=bundle.support_envelope.contract_version,
        support_envelope_fingerprint=bundle.support_envelope.artifact.sha256,
        bundle_id=bundle.bundle_id,
        bundle_version=bundle.bundle_version,
        bundle_fingerprint=bundle.fingerprint,
        bundle_kind=bundle.bundle_kind,
        targets=(target,),
        required_ports=(BiologicalStagePort.ACTION_CONTEXT_ENCODER,),
        envelope_missing_ports=(),
        envelope_extra_ports=(),
        invalid_dispositions=(),
        scope_issues=(),
    )


def _jit_context(*, report=None, runtime_interfaces=None, jit_loaders=None):
    bundle, _, receipt_batch = admission_cases._complete_report()
    verifier = admission_cases._trusted_verifier()
    loader = admission_cases._jit_loader_with_verifier(verifier)
    return AdmissionVerificationContext(
        receipt_batch=receipt_batch if report is None else report,
        validation_result_batch=None,
        query_prerequisite_report=_jit_report(bundle),
        trusted_verifiers=(verifier,),
        runtime_interfaces=(
            admission_cases._runtime_interfaces()
            if runtime_interfaces is None
            else runtime_interfaces
        ),
        jit_loaders={loader.fingerprint: loader} if jit_loaders is None else jit_loaders,
    )


def test_context_and_jit_select_only_the_exact_operation_surface() -> None:
    context = _jit_context()
    handles = reverify_admission_jit_interfaces(
        context,
        provider=lambda _receipt: admission_cases.CODE_BYTES,
        operation=ModelOperation.ESTIMATE_CELL_STATE,
    )

    assert {(handle.target_kind, handle.target_id) for handle in handles} == {
        (ImplementationReceiptTargetKind.PORT, BiologicalStagePort.ACTION_CONTEXT_ENCODER.value),
        (
            ImplementationReceiptTargetKind.RUNTIME_OPERATION,
            ModelOperation.ESTIMATE_CELL_STATE.value,
        ),
    }
    assert all(handle.loaded_object is admission_cases.GoodImplementation for handle in handles)

    with pytest.raises(ValueError, match="one exact query-prerequisite target"):
        reverify_admission_jit_interfaces(
            context,
            provider=lambda _receipt: admission_cases.CODE_BYTES,
            operation=ModelOperation.EVOLVE_CELL_STATE,
        )

    missing_operation = context.receipt_batch.model_copy(
        update={
            "interface_receipts": tuple(
                receipt
                for receipt in context.receipt_batch.interface_receipts
                if receipt.requirement.target_kind is ImplementationReceiptTargetKind.PORT
            )
        }
    )
    with pytest.raises(ValueError, match="omit an authorized prerequisite target"):
        reverify_admission_jit_interfaces(
            _jit_context(report=missing_operation),
            provider=lambda _receipt: admission_cases.CODE_BYTES,
            operation=ModelOperation.ESTIMATE_CELL_STATE,
        )

    repeated = context.receipt_batch.model_copy(
        update={
            "interface_receipts": (
                context.receipt_batch.interface_receipts[0],
                *context.receipt_batch.interface_receipts,
            )
        }
    )
    with pytest.raises(ValueError, match="repeat one implementation target"):
        reverify_admission_jit_interfaces(
            _jit_context(report=repeated),
            provider=lambda _receipt: admission_cases.CODE_BYTES,
            operation=ModelOperation.ESTIMATE_CELL_STATE,
        )

    with pytest.raises(ValueError, match="trusted runtime interface"):
        reverify_admission_jit_interfaces(
            _jit_context(runtime_interfaces={}),
            provider=lambda _receipt: admission_cases.CODE_BYTES,
            operation=ModelOperation.ESTIMATE_CELL_STATE,
        )
    with pytest.raises(ValueError, match="trusted isolated loader"):
        reverify_admission_jit_interfaces(
            _jit_context(jit_loaders={}),
            provider=lambda _receipt: admission_cases.CODE_BYTES,
            operation=ModelOperation.ESTIMATE_CELL_STATE,
        )


def test_context_rejects_untrusted_or_misindexed_runtime_authority() -> None:
    bundle, _, receipt_batch = admission_cases._complete_report()
    verifier = admission_cases._trusted_verifier()
    loader = admission_cases._jit_loader_with_verifier(verifier)
    common = {
        "receipt_batch": receipt_batch,
        "validation_result_batch": None,
        "query_prerequisite_report": _jit_report(bundle),
        "runtime_interfaces": admission_cases._runtime_interfaces(),
    }

    with pytest.raises(ValueError, match="external verifier trust root"):
        AdmissionVerificationContext(**common, trusted_verifiers=())
    with pytest.raises(ValueError, match="identity/key pairs must be unique"):
        AdmissionVerificationContext(**common, trusted_verifiers=(verifier, verifier))
    with pytest.raises(ValueError, match="registry keys must match"):
        AdmissionVerificationContext(
            **{
                **common,
                "runtime_interfaces": {"wrong.Interface": admission_cases._trusted_interface()},
            },
            trusted_verifiers=(verifier,),
        )
    with pytest.raises(ValueError, match="registry key differs"):
        AdmissionVerificationContext(
            **common,
            trusted_verifiers=(verifier,),
            jit_loaders={"0" * 64: loader},
        )

    other_identity = AdmissionVerifierIdentity.model_validate(
        {
            **verifier.identity.model_dump(mode="python"),
            "verifier_version": "0.2.0",
        }
    )
    other_verifier = TrustedAdmissionVerifier(
        identity=other_identity,
        key_id="other-loader-key",
        secret=b"o" * 32,
    )
    other_loader = TrustedJITLoader(
        verifier=other_verifier,
        load_exact=lambda _receipt, _code: admission_cases.GoodImplementation,
    )
    with pytest.raises(ValueError, match="absent from the external trust root"):
        AdmissionVerificationContext(
            **common,
            trusted_verifiers=(verifier,),
            jit_loaders={other_loader.fingerprint: other_loader},
        )


@pytest.mark.parametrize(
    ("outcome", "evidence", "reasons", "message"),
    (
        (CriterionOutcome.PASSED, (), (), "passed validation criterion requires evidence"),
        (CriterionOutcome.FAILED, ("result",), (), "failed validation criterion requires evidence"),
        (CriterionOutcome.NOT_EVALUATED, ("result",), ("blocked",), "require reasons and no"),
    ),
)
def test_validation_criterion_evidence_is_fail_closed(
    outcome: CriterionOutcome,
    evidence: tuple[str, ...],
    reasons: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ValidationCriterionResult(
            criterion=validation_cases.ValidationSemanticCriterion.RESULT_SCHEMA_CONFORMANCE,
            outcome=outcome,
            evidence_artifact_ids=evidence,
            reasons=reasons,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("stale_policy", "stale semantic-criteria policy"),
        ("self_reference", "cannot include its own"),
        ("unknown_citation", "undeclared supporting artifacts"),
        ("unused_support", "every validation supporting artifact"),
        ("non_utc", "timezone-aware UTC"),
    ),
)
def test_manifest_revalidation_closes_model_copy_bypasses(
    mutation: str,
    message: str,
    validation_scenario,
) -> None:
    manifest = validation_scenario.prepared.manifest
    payload = manifest.model_dump(mode="python")
    if mutation == "stale_policy":
        payload["criteria_policy_fingerprint"] = "0" * 64
    elif mutation == "self_reference":
        payload["supporting_artifacts"] = (
            validation_cases._artifact(manifest.result_manifest_artifact_id, b"self"),
        )
    elif mutation == "unknown_citation":
        criteria = list(payload["criteria"])
        criteria[0] = {**criteria[0], "evidence_artifact_ids": ("unknown-result",)}
        payload["criteria"] = tuple(criteria)
    elif mutation == "unused_support":
        payload["supporting_artifacts"] = (
            *payload["supporting_artifacts"],
            validation_cases._artifact("zz-unused-result", b"unused"),
        )
    else:
        payload["generated_at"] = manifest.generated_at.replace(tzinfo=None)

    with pytest.raises(ValidationError, match=message):
        ValidationResultManifest.model_validate(payload)

    assert len(manifest.fingerprint) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("media_type", "exact media type"),
        ("container_id", "name its containing artifact"),
        ("scope", "scope must match"),
        ("coverage", "stale artifact coverage"),
        ("evaluator", "fingerprint does not match"),
        ("capability", "lacks semantic-result capability"),
        ("isolation", "distinct from result artifacts"),
        ("attestation", "binds a different observation"),
        ("fingerprint", "fingerprint does not match"),
    ),
)
def test_semantic_receipt_model_copy_is_revalidated_at_the_boundary(
    mutation: str,
    message: str,
    validation_scenario,
) -> None:
    receipt = validation_scenario.semantic_receipt
    update = {}
    if mutation == "media_type":
        update["result_manifest_artifact"] = receipt.result_manifest_artifact.model_copy(
            update={"media_type": "application/json"}
        )
    elif mutation == "container_id":
        update["result_manifest"] = receipt.result_manifest.model_copy(
            update={"result_manifest_artifact_id": "other-manifest"}
        )
    elif mutation == "scope":
        update["validation_scope_fingerprint"] = "0" * 64
    elif mutation == "coverage":
        update["artifact_coverage_fingerprint"] = "0" * 64
    elif mutation == "evaluator":
        update["evaluator_fingerprint"] = "0" * 64
    elif mutation == "capability":
        evaluator = AdmissionVerifierIdentity.model_validate(
            {
                **receipt.evaluator.model_dump(mode="python"),
                "capabilities": (AdmissionVerifierCapability.ARTIFACT_BYTE_RESOLUTION,),
            }
        )
        update.update(evaluator=evaluator, evaluator_fingerprint=evaluator.fingerprint)
    elif mutation == "isolation":
        update["isolation_evidence_artifact"] = receipt.result_manifest.supporting_artifacts[0]
    elif mutation == "attestation":
        update["attestation"] = receipt.attestation.model_copy(
            update={"attested_payload_fingerprint": "0" * 64}
        )
    else:
        update["receipt_fingerprint"] = "0" * 64

    bypass = receipt.model_copy(update=update)
    with pytest.raises(ValidationError, match=message):
        ValidationResultVerificationReceipt.model_validate(bypass.model_dump(mode="python"))


def test_execution_artifact_collection_is_recursive_deduplicated_and_conflict_safe(
    validation_scenario,
) -> None:
    query = StateQuery.model_validate_json(validation_cases.QUERY_PATH.read_bytes())
    benchmark = BenchmarkArtifact.model_validate_json(validation_cases.BENCHMARK_PATH.read_bytes())
    _, training, _, _ = complete_looking_runtime_contracts(query, benchmark)
    artifacts = admission_execution_artifacts(
        benchmark=benchmark,
        training_run=training,
        validation_evidence=(validation_scenario.prepared.evidence,),
    )
    artifact_ids = tuple(artifact.artifact_id for artifact in artifacts)

    assert artifact_ids == tuple(sorted(set(artifact_ids)))
    assert training.model_artifact.artifact_id in artifact_ids
    assert {
        artifact.artifact_id
        for artifact in validation_scenario.prepared.evidence.evidence_artifacts
    }.issubset(artifact_ids)

    first = validation_cases._artifact("conflicting-result", b"first")
    second = validation_cases._artifact("conflicting-result", b"second")
    copied_evidence = validation_scenario.prepared.evidence.model_copy(
        update={"evidence_artifacts": (first, second)}
    )
    with pytest.raises(ValueError, match="conflicting byte declarations"):
        admission_execution_artifacts(
            benchmark=benchmark,
            training_run=None,
            validation_evidence=(copied_evidence,),
        )


def test_receipt_revalidation_rejects_future_artifact_receipts(validation_scenario) -> None:
    receipt = validation_scenario.semantic_receipt
    artifact_receipt = receipt.artifact_receipts[0]
    future = artifact_receipt.model_copy(
        update={"issued_at": receipt.issued_at + timedelta(seconds=1)}
    )
    copied = receipt.model_copy(
        update={"artifact_receipts": (future, *receipt.artifact_receipts[1:])}
    )

    with pytest.raises(ValidationError):
        ValidationResultVerificationReceipt.model_validate(copied.model_dump(mode="python"))
