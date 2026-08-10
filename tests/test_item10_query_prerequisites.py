"""Exact, deterministic query-derived biological-port prerequisite tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import environment_spec_factory, query_factory

from cellstate.backends.contracts import (
    OPERATION_REQUIRED_PORTS,
    QUERY_PREREQUISITE_COMPILER_FINGERPRINT,
    QUERY_PREREQUISITE_COMPILER_ID,
    QUERY_PREREQUISITE_COMPILER_VERSION,
    BiologicalModelBundleContract,
    BiologicalStagePort,
    BiologicalSupportEnvelope,
    BundleContractKind,
    BundleContractReference,
    DirectPopulationResponseSemantics,
    ModelOperation,
    ModelPortBinding,
    PortDisposition,
    QueryDerivedPrerequisiteReport,
    QueryPrerequisiteReasonCode,
    QueryPrerequisiteScopeIssue,
    ValidationEvidenceKind,
    ValidationEvidenceRequirement,
    assess_biological_model_bundle,
    derive_query_prerequisite_report,
    verify_query_prerequisite_report,
)
from cellstate.data import (
    BenchmarkArtifact,
    BenchmarkPartitionRole,
    ContentAddressedArtifact,
    DatasetManifest,
)
from cellstate.domain import StateQuery, SystemBoundary
from cellstate.domain.common import SchemaModel, canonical_fingerprint, canonical_json_bytes
from cellstate.domain.events import EvidenceRole

ROOT = Path(__file__).resolve().parents[1]
QUERY_PATH = ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/state-query.json"
BENCHMARK_PATH = ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/benchmark-artifact.json"
MANIFEST_PATH = ROOT / "data_manifests/reviewed/sciplex3-k562-24h.json"
COMPONENT_ENVELOPE_PATH = ROOT / "backends/vertical-a/sciplex3-k562-24h-v1/support-envelope.json"
COMPONENT_BUNDLE_PATH = ROOT / "backends/vertical-a/sciplex3-k562-24h-v1/bundle-contract.json"


def _reference(
    contract_id: str,
    contract_version: str,
    payload: SchemaModel,
) -> BundleContractReference:
    content = canonical_json_bytes(payload.model_dump(mode="json"))
    return BundleContractReference(
        contract_id=contract_id,
        contract_version=contract_version,
        artifact=ContentAddressedArtifact(
            artifact_id=f"{contract_id}-canonical-json",
            uri=f"https://example.invalid/contracts/{contract_id}.json",
            sha256=canonical_fingerprint(payload.model_dump(mode="json")),
            byte_count=len(content),
            media_type="application/json",
        ),
    )


def _ports(
    required: set[BiologicalStagePort],
    *,
    unavailable: BiologicalStagePort | None = None,
    unavailable_disposition: PortDisposition = PortDisposition.UNSUPPORTED,
) -> tuple[ModelPortBinding, ...]:
    bindings = []
    for port in BiologicalStagePort:
        disposition = (
            PortDisposition.REQUIRED if port in required else PortDisposition.NOT_APPLICABLE
        )
        if port is unavailable:
            disposition = unavailable_disposition
        bindings.append(
            ModelPortBinding(
                port=port,
                disposition=disposition,
                rationale=("Exact query-prerequisite test classification.",),
            )
        )
    return tuple(sorted(bindings, key=lambda item: item.port.value))


def _runtime_contracts(
    query: StateQuery,
    operation: ModelOperation,
    *,
    benchmark: BenchmarkArtifact | None = None,
    unavailable: BiologicalStagePort | None = None,
    unavailable_disposition: PortDisposition = PortDisposition.UNSUPPORTED,
) -> tuple[BiologicalSupportEnvelope, BiologicalModelBundleContract]:
    query_ref = (
        _reference("test-query", "1", query)
        if benchmark is None
        else _reference(
            benchmark.definition.query.query_id,
            benchmark.definition.query.query_version,
            query,
        )
    )
    benchmark_payload: SchemaModel = query if benchmark is None else benchmark
    benchmark_ref = _reference(
        "test-benchmark" if benchmark is None else benchmark.definition.benchmark_id,
        "1" if benchmark is None else benchmark.definition.benchmark_version,
        benchmark_payload,
    )

    def build_envelope(
        required: tuple[BiologicalStagePort, ...],
    ) -> BiologicalSupportEnvelope:
        return BiologicalSupportEnvelope(
            envelope_id=f"test-{operation.value}-envelope",
            envelope_version="0.1.0",
            bundle_kind=BundleContractKind.BIOLOGICAL_MODEL_BUNDLE,
            query=query_ref,
            benchmark=benchmark_ref,
            runtime_operations=(operation,),
            required_ports=required,
            required_validation_evidence=(
                ValidationEvidenceRequirement(
                    evidence_id="runtime-validation",
                    evidence_kind=ValidationEvidenceKind.RUNTIME_OPERATION_VALIDATION,
                    partition_roles=(BenchmarkPartitionRole.UNTOUCHED_TEST,),
                ),
            ),
            notes=("Test-only runtime prerequisite envelope.",),
        )

    floor = tuple(sorted(OPERATION_REQUIRED_PORTS[operation], key=lambda item: item.value))
    provisional_envelope = build_envelope(floor)
    provisional_bundle = BiologicalModelBundleContract(
        bundle_id=f"test-{operation.value}-bundle",
        bundle_version="0.1.0",
        bundle_kind=BundleContractKind.BIOLOGICAL_MODEL_BUNDLE,
        description="Test-only query-prerequisite bundle.",
        posterior_schema_id="cellstate.v2.test-posterior",
        query=query_ref,
        benchmark=benchmark_ref,
        support_envelope=_reference(
            provisional_envelope.envelope_id,
            provisional_envelope.envelope_version,
            provisional_envelope,
        ),
        ports=_ports(set(floor)),
    )
    provisional_report = derive_query_prerequisite_report(
        query=query,
        support_envelope=provisional_envelope,
        bundle=provisional_bundle,
    )
    envelope = build_envelope(provisional_report.required_ports)
    bundle = BiologicalModelBundleContract(
        bundle_id=provisional_bundle.bundle_id,
        bundle_version=provisional_bundle.bundle_version,
        bundle_kind=BundleContractKind.BIOLOGICAL_MODEL_BUNDLE,
        description=provisional_bundle.description,
        posterior_schema_id=provisional_bundle.posterior_schema_id,
        query=query_ref,
        benchmark=benchmark_ref,
        support_envelope=_reference(
            envelope.envelope_id,
            envelope.envelope_version,
            envelope,
        ),
        ports=_ports(
            set(provisional_report.required_ports),
            unavailable=unavailable,
            unavailable_disposition=unavailable_disposition,
        ),
    )
    return envelope, bundle


def test_direct_component_prerequisites_are_explicit_and_exact() -> None:
    query = StateQuery.model_validate_json(QUERY_PATH.read_bytes())
    envelope = BiologicalSupportEnvelope.model_validate_json(COMPONENT_ENVELOPE_PATH.read_bytes())
    bundle = BiologicalModelBundleContract.model_validate_json(COMPONENT_BUNDLE_PATH.read_bytes())

    report = derive_query_prerequisite_report(
        query=query,
        support_envelope=envelope,
        bundle=bundle,
    )

    assert report.compiler_id == QUERY_PREREQUISITE_COMPILER_ID
    assert report.compiler_version == QUERY_PREREQUISITE_COMPILER_VERSION
    assert report.compiler_fingerprint == QUERY_PREREQUISITE_COMPILER_FINGERPRINT
    assert report.query_fingerprint == query.fingerprint
    assert report.support_envelope_fingerprint == envelope.fingerprint
    assert report.bundle_fingerprint == bundle.fingerprint
    assert report.structurally_satisfied
    assert report.required_ports == envelope.required_ports
    assert len(report.targets) == 1
    target = report.targets[0]
    assert target.operation is None
    assert target.direct_population_response == DirectPopulationResponseSemantics()
    assert {prerequisite.port: set(prerequisite.reasons) for prerequisite in target.required_ports}[
        BiologicalStagePort.POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL
    ] == {QueryPrerequisiteReasonCode.COMPONENT_RESPONSE_DISTRIBUTION}
    assert (
        verify_query_prerequisite_report(
            report,
            query=query,
            support_envelope=envelope,
            bundle=bundle,
        )
        is not report
    )


def test_runtime_derivation_covers_query_conditionals_and_machine_reasons() -> None:
    payload = query_factory().model_dump(mode="python")
    payload["system_boundary"] = SystemBoundary.SPATIAL_TISSUE_NICHE
    payload["evidence_policy"]["allowed_evidence_roles"] = (
        EvidenceRole.DIRECT,
        EvidenceRole.ANCESTOR,
        EvidenceRole.SPATIAL_NEIGHBOR,
    )
    payload["constraints"]["allow_transport"] = True
    payload["constraints"]["require_complete_lineage_history"] = True
    payload["constraints"]["require_complete_neighborhood_history"] = True
    query = StateQuery.model_validate(payload)
    envelope, bundle = _runtime_contracts(
        query,
        ModelOperation.RECOMMEND_NEXT_MEASUREMENT,
    )

    first = derive_query_prerequisite_report(
        query=query,
        support_envelope=envelope,
        bundle=bundle,
    )
    second = derive_query_prerequisite_report(
        query=query,
        support_envelope=envelope,
        bundle=bundle,
    )
    required = set(first.required_ports)

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert first.structurally_satisfied
    assert {
        BiologicalStagePort.CELL_INTERACTION_MODEL,
        BiologicalStagePort.DIVISION_AND_INHERITANCE_MODEL,
        BiologicalStagePort.EVIDENCE_TRANSFER_MODELS,
        BiologicalStagePort.EXTRACELLULAR_TRANSPORT_MODEL,
        BiologicalStagePort.IDENTIFIABILITY_ANALYZER,
        BiologicalStagePort.INTERVENTION_REALIZATION_MODEL,
        BiologicalStagePort.MECHANISTIC_CONSTRAINTS,
        BiologicalStagePort.MODEL_ENSEMBLE,
        BiologicalStagePort.OBSERVATION_MODELS,
        BiologicalStagePort.POSTERIOR_INFERENCE_ENGINE,
        BiologicalStagePort.SUFFICIENCY_EVALUATOR,
        BiologicalStagePort.TRANSITION_MODEL,
        BiologicalStagePort.VALUE_OF_INFORMATION_ENGINE,
    } <= required
    by_port = {
        prerequisite.port: set(prerequisite.reasons)
        for prerequisite in first.targets[0].required_ports
    }
    assert (
        QueryPrerequisiteReasonCode.NONDIRECT_EVIDENCE_TRANSFER
        in by_port[BiologicalStagePort.EVIDENCE_TRANSFER_MODELS]
    )
    assert (
        QueryPrerequisiteReasonCode.HYPOTHETICAL_MEASUREMENT_UPDATE
        in by_port[BiologicalStagePort.POSTERIOR_INFERENCE_ENGINE]
    )
    assert (
        QueryPrerequisiteReasonCode.EPISTEMIC_MODEL_UNCERTAINTY
        in by_port[BiologicalStagePort.MODEL_ENSEMBLE]
    )
    assert (
        QueryPrerequisiteReasonCode.COUNTERFACTUAL_REPLANNING
        in by_port[BiologicalStagePort.TRANSITION_MODEL]
    )


@pytest.mark.parametrize(
    ("operation", "operation_specific"),
    (
        (
            ModelOperation.ESTIMATE_CELL_STATE,
            {
                BiologicalStagePort.INTERVENTION_REALIZATION_MODEL,
                BiologicalStagePort.OBSERVATION_MODELS,
                BiologicalStagePort.POSTERIOR_INFERENCE_ENGINE,
                BiologicalStagePort.TRANSITION_MODEL,
            },
        ),
        (
            ModelOperation.EVOLVE_CELL_STATE,
            {
                BiologicalStagePort.FUNCTIONAL_DECODERS,
                BiologicalStagePort.INTERVENTION_REALIZATION_MODEL,
                BiologicalStagePort.TRANSITION_MODEL,
            },
        ),
        (
            ModelOperation.CHOOSE_INTERVENTION,
            {
                BiologicalStagePort.IDENTIFIABILITY_ANALYZER,
                BiologicalStagePort.INTERVENTION_REALIZATION_MODEL,
                BiologicalStagePort.SUFFICIENCY_EVALUATOR,
                BiologicalStagePort.TRANSITION_MODEL,
            },
        ),
        (
            ModelOperation.RECOMMEND_NEXT_MEASUREMENT,
            {
                BiologicalStagePort.IDENTIFIABILITY_ANALYZER,
                BiologicalStagePort.OBSERVATION_MODELS,
                BiologicalStagePort.POSTERIOR_INFERENCE_ENGINE,
                BiologicalStagePort.SUFFICIENCY_EVALUATOR,
                BiologicalStagePort.VALUE_OF_INFORMATION_ENGINE,
            },
        ),
    ),
)
def test_each_public_operation_combines_floor_and_query_prerequisites(
    operation: ModelOperation,
    operation_specific: set[BiologicalStagePort],
) -> None:
    query = query_factory()
    envelope, bundle = _runtime_contracts(query, operation)
    report = derive_query_prerequisite_report(
        query=query,
        support_envelope=envelope,
        bundle=bundle,
    )
    required = set(report.required_ports)

    assert set(OPERATION_REQUIRED_PORTS[operation]) <= required
    assert BiologicalStagePort.MODEL_ENSEMBLE in required
    assert BiologicalStagePort.FUNCTIONAL_DECODERS in required
    assert operation_specific <= required
    assert report.structurally_satisfied
    assert not report.scope_issues


def test_environment_and_soluble_boundary_derive_dynamics_and_transport() -> None:
    payload = query_factory().model_dump(mode="python")
    payload["system_boundary"] = SystemBoundary.CELL_AND_SOLUBLE_ENVIRONMENT
    payload["environment_space"] = (environment_spec_factory(),)
    payload["constraints"]["require_complete_environment_history"] = True
    query = StateQuery.model_validate(payload)
    envelope, bundle = _runtime_contracts(query, ModelOperation.ESTIMATE_CELL_STATE)
    report = derive_query_prerequisite_report(
        query=query,
        support_envelope=envelope,
        bundle=bundle,
    )
    by_port = {
        prerequisite.port: set(prerequisite.reasons)
        for prerequisite in report.targets[0].required_ports
    }

    assert (
        QueryPrerequisiteReasonCode.ENVIRONMENT_HISTORY_DYNAMICS
        in by_port[BiologicalStagePort.TRANSITION_MODEL]
    )
    assert (
        QueryPrerequisiteReasonCode.SOLUBLE_OR_SPATIAL_TRANSPORT
        in by_port[BiologicalStagePort.EXTRACELLULAR_TRANSPORT_MODEL]
    )


def test_recommendation_without_actions_or_candidate_assays_has_scope_issues() -> None:
    payload = query_factory().model_dump(mode="python")
    payload["intervention_space"] = ()
    payload["available_assays"] = ()
    payload["constraints"].update(
        {
            "maximum_total_assay_cost": None,
            "assay_cost_units": None,
            "maximum_assay_delay_seconds": None,
        }
    )
    query = StateQuery.model_validate(payload)
    envelope, bundle = _runtime_contracts(
        query,
        ModelOperation.RECOMMEND_NEXT_MEASUREMENT,
    )
    report = derive_query_prerequisite_report(
        query=query,
        support_envelope=envelope,
        bundle=bundle,
    )

    assert set(report.scope_issues) == {
        QueryPrerequisiteScopeIssue.MEASUREMENT_EVSI_REQUIRES_ACTION_SPACE,
        QueryPrerequisiteScopeIssue.MEASUREMENT_SELECTION_REQUIRES_CANDIDATE_ASSAYS,
    }
    assert not report.structurally_satisfied


@pytest.mark.parametrize(
    "unavailable_disposition",
    (
        PortDisposition.NOT_APPLICABLE,
        PortDisposition.PLANNED,
        PortDisposition.UNSUPPORTED,
    ),
)
def test_unavailable_derived_port_fails_closed_and_is_an_assessment_blocker(
    unavailable_disposition: PortDisposition,
) -> None:
    query = StateQuery.model_validate_json(QUERY_PATH.read_bytes())
    benchmark = BenchmarkArtifact.model_validate_json(BENCHMARK_PATH.read_bytes())
    manifest = DatasetManifest.model_validate_json(MANIFEST_PATH.read_bytes())
    envelope, bundle = _runtime_contracts(
        query,
        ModelOperation.EVOLVE_CELL_STATE,
        benchmark=benchmark,
        unavailable=BiologicalStagePort.MODEL_ENSEMBLE,
        unavailable_disposition=unavailable_disposition,
    )
    report = derive_query_prerequisite_report(
        query=query,
        support_envelope=envelope,
        bundle=bundle,
    )

    assert not report.structurally_satisfied
    assert [(failure.port, failure.disposition) for failure in report.invalid_dispositions] == [
        (BiologicalStagePort.MODEL_ENSEMBLE, unavailable_disposition)
    ]
    readiness = assess_biological_model_bundle(
        bundle,
        query=query,
        benchmark=benchmark,
        manifests={
            binding.binding_id: manifest for binding in benchmark.definition.evidence_bindings
        },
        support_envelope=envelope,
    )
    assert readiness.query_derived_prerequisite_report == report
    assert (
        readiness.query_derived_prerequisite_fingerprint
        == readiness.query_derived_prerequisite_report.fingerprint
    )
    assert not readiness.query_derived_prerequisites_verified
    assert (
        f"query-derived prerequisite port model_ensemble is {unavailable_disposition.value}"
        in readiness.blockers
    )


def test_verifier_rederives_and_rejects_report_or_compiler_identity_drift() -> None:
    query = query_factory()
    envelope, bundle = _runtime_contracts(query, ModelOperation.ESTIMATE_CELL_STATE)
    report = derive_query_prerequisite_report(
        query=query,
        support_envelope=envelope,
        bundle=bundle,
    )
    payload = report.model_dump(mode="python")
    payload["bundle_version"] = "forged-version"
    forged = QueryDerivedPrerequisiteReport.model_validate(payload)

    with pytest.raises(ValueError, match="does not match exact"):
        verify_query_prerequisite_report(
            forged,
            query=query,
            support_envelope=envelope,
            bundle=bundle,
        )

    payload = report.model_dump(mode="python")
    payload["compiler_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="canonical compiler identity"):
        QueryDerivedPrerequisiteReport.model_validate(payload)
