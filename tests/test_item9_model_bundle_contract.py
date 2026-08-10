"""Fail-closed tests for the biological bundle and support-port contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from cellstate.backends import (
    OPERATION_REQUIRED_PORTS,
    BiologicalExecutionBlockedError,
    BiologicalModelBundleContract,
    BiologicalStagePort,
    BiologicalSupportEnvelope,
    BundleAdmissionBlockerCode,
    BundleContractKind,
    BundleContractReference,
    ComponentLifecycleStage,
    DirectPopulationResponseSemantics,
    ModelOperation,
    ModelOperationImplementationBinding,
    ModelPortBinding,
    PortDisposition,
    PortImplementationBinding,
    PortImplementationKind,
    TrainingRunBinding,
    ValidationEvidenceBinding,
    ValidationEvidenceKind,
    ValidationEvidenceRequirement,
    assess_biological_model_bundle,
    build_admitted_estimator_descriptor,
    require_biological_component_execution,
    require_biological_execution,
)
from cellstate.data import (
    BenchmarkArtifact,
    BenchmarkPartitionRole,
    ContentAddressedArtifact,
    DatasetManifest,
)
from cellstate.domain import StateQuery
from cellstate.domain.common import SchemaModel, canonical_fingerprint, canonical_json_bytes

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/benchmark-artifact.json"
MANIFEST_PATH = ROOT / "data_manifests/reviewed/sciplex3-k562-24h.json"
QUERY_PATH = ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/state-query.json"

COMPONENT_REQUIRED_PORTS = tuple(
    sorted(
        (
            BiologicalStagePort.ACTION_CONTEXT_ENCODER,
            BiologicalStagePort.ARTIFACT_PROVENANCE_WRITER,
            BiologicalStagePort.EXACT_ARTIFACT_RESOLVER,
            BiologicalStagePort.POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL,
            BiologicalStagePort.QUERY_SCOPE_VALIDATOR,
            BiologicalStagePort.STRICT_SUPPORT_OOD_GATE,
            BiologicalStagePort.TRAIN_CAL_DATA_LOADER,
            BiologicalStagePort.UNCERTAINTY_CALIBRATOR,
        ),
        key=lambda port: port.value,
    )
)


@pytest.fixture(scope="module")
def query() -> StateQuery:
    return StateQuery.model_validate_json(QUERY_PATH.read_bytes())


@pytest.fixture(scope="module")
def benchmark() -> BenchmarkArtifact:
    return BenchmarkArtifact.model_validate_json(BENCHMARK_PATH.read_bytes())


@pytest.fixture(scope="module")
def manifest() -> DatasetManifest:
    return DatasetManifest.model_validate_json(MANIFEST_PATH.read_bytes())


def contract_reference(
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
            uri=f"https://example.invalid/contracts/{contract_id}/{contract_version}.json",
            sha256=canonical_fingerprint(payload.model_dump(mode="json")),
            byte_count=len(content),
            media_type="application/json",
        ),
    )


def code_artifact(port: BiologicalStagePort) -> ContentAddressedArtifact:
    return ContentAddressedArtifact(
        artifact_id=f"{port.value}-code",
        uri=f"https://example.invalid/code/{port.value}.py",
        sha256=canonical_fingerprint({"port": port.value}),
        byte_count=1,
        media_type="text/x-python",
    )


def named_artifact(artifact_id: str) -> ContentAddressedArtifact:
    return ContentAddressedArtifact(
        artifact_id=artifact_id,
        uri=f"https://example.invalid/artifacts/{artifact_id}",
        sha256=canonical_fingerprint({"artifact_id": artifact_id}),
        byte_count=1,
        media_type="application/octet-stream",
    )


def port_map(
    required_ports: tuple[BiologicalStagePort, ...],
    *,
    required_disposition: PortDisposition = PortDisposition.REQUIRED,
    implementation_kind: PortImplementationKind = PortImplementationKind.PYTHON_ENTRY_POINT,
    evidence_id: str = "component-validation",
) -> tuple[ModelPortBinding, ...]:
    bindings = []
    required = set(required_ports)
    for port in BiologicalStagePort:
        if port not in required:
            bindings.append(
                ModelPortBinding(
                    port=port,
                    disposition=PortDisposition.NOT_APPLICABLE,
                    rationale=("Outside this exact endpoint-component scope.",),
                )
            )
            continue
        implementation = None
        evidence_ids: tuple[str, ...] = ()
        if required_disposition is PortDisposition.PROVIDED:
            implementation = PortImplementationBinding(
                implementation_id=f"cellstate.{port.value}",
                implementation_version="0.1.0",
                interface=f"cellstate.backends.{port.value}",
                kind=implementation_kind,
                code_artifact=code_artifact(port),
                entrypoint=(
                    f"cellstate.backends.population_response:{port.value}"
                    if implementation_kind is PortImplementationKind.PYTHON_ENTRY_POINT
                    else None
                ),
            )
            evidence_ids = (evidence_id,)
        bindings.append(
            ModelPortBinding(
                port=port,
                disposition=required_disposition,
                implementation=implementation,
                validation_evidence_ids=evidence_ids,
                rationale=("Required by the exact component support envelope.",),
            )
        )
    return tuple(sorted(bindings, key=lambda binding: binding.port.value))


def component_contracts(
    query: StateQuery,
    benchmark: BenchmarkArtifact,
) -> tuple[BiologicalSupportEnvelope, BiologicalModelBundleContract]:
    query_ref = contract_reference(
        benchmark.definition.query.query_id,
        benchmark.definition.query.query_version,
        query,
    )
    benchmark_ref = contract_reference(
        benchmark.definition.benchmark_id,
        benchmark.definition.benchmark_version,
        benchmark,
    )
    envelope = BiologicalSupportEnvelope(
        envelope_id="sciplex3-k562-direct-response-envelope",
        envelope_version="0.1.0",
        bundle_kind=BundleContractKind.COMPONENT_SCAFFOLD,
        query=query_ref,
        benchmark=benchmark_ref,
        direct_population_response=DirectPopulationResponseSemantics(),
        runtime_operations=(),
        required_ports=COMPONENT_REQUIRED_PORTS,
        required_validation_evidence=(
            ValidationEvidenceRequirement(
                evidence_id="component-validation",
                evidence_kind=ValidationEvidenceKind.LOCKED_COMPONENT_EVALUATION,
                partition_roles=(BenchmarkPartitionRole.UNTOUCHED_TEST,),
            ),
        ),
        notes=(
            "The 24-hour endpoint and matched vehicles are targets/comparators, never t=0 priors.",
        ),
    )
    bundle = BiologicalModelBundleContract(
        bundle_id="sciplex3-k562-direct-population-response",
        bundle_version="0.1.0-scaffold",
        bundle_kind=BundleContractKind.COMPONENT_SCAFFOLD,
        description="Non-runnable direct context-to-endpoint population response scaffold.",
        query=query_ref,
        benchmark=benchmark_ref,
        support_envelope=contract_reference(
            envelope.envelope_id,
            envelope.envelope_version,
            envelope,
        ),
        ports=port_map(COMPONENT_REQUIRED_PORTS),
    )
    return envelope, bundle


def test_component_scaffold_is_exact_but_cannot_register_or_execute(
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    manifest: DatasetManifest,
) -> None:
    envelope, bundle = component_contracts(query, benchmark)
    readiness = assess_biological_model_bundle(
        bundle,
        query=query,
        benchmark=benchmark,
        manifests={
            binding.binding_id: manifest for binding in benchmark.definition.evidence_bindings
        },
        support_envelope=envelope,
    )

    assert readiness.query_binding_verified
    assert readiness.lifecycle_stage is ComponentLifecycleStage.SCAFFOLD
    assert readiness.benchmark_binding_verified
    assert readiness.support_envelope_binding_verified
    assert not readiness.training_binding_verified
    assert not readiness.calibration_binding_verified
    assert not readiness.model_selection_binding_verified
    assert not readiness.validation_bindings_verified
    assert not readiness.required_ports_provided
    assert not readiness.required_ports_executable
    assert not readiness.required_ports_evidenced
    assert not readiness.benchmark_admission_ready
    assert not readiness.component_evaluation_complete
    assert not readiness.component_executable
    assert not readiness.scientifically_admitted
    assert "one or more public runtime operations are specification-only" not in readiness.blockers
    assert not readiness.runtime_registration_allowed
    assert not readiness.runnable
    assert "bound benchmark is not scientifically admitted" in readiness.blockers
    assert "component scaffold is not a public cell-state runtime backend" in readiness.blockers

    kwargs = {
        "query": query,
        "benchmark": benchmark,
        "manifests": {
            binding.binding_id: manifest for binding in benchmark.definition.evidence_bindings
        },
        "support_envelope": envelope,
    }
    with pytest.raises(BiologicalExecutionBlockedError):
        require_biological_execution(
            bundle,
            operation=ModelOperation.ESTIMATE_CELL_STATE,
            **kwargs,
        )
    with pytest.raises(BiologicalExecutionBlockedError):
        require_biological_component_execution(bundle, **kwargs)
    with pytest.raises(BiologicalExecutionBlockedError):
        build_admitted_estimator_descriptor(bundle, **kwargs)


def test_port_map_is_exhaustive_and_preserves_original_skeleton(
    query: StateQuery,
    benchmark: BenchmarkArtifact,
) -> None:
    _, bundle = component_contracts(query, benchmark)
    original_names = {
        "cell_interaction_model",
        "division_and_inheritance_model",
        "evidence_transfer_models",
        "extracellular_transport_model",
        "functional_decoders",
        "identifiability_analyzer",
        "intervention_realization_model",
        "mechanistic_constraints",
        "model_ensemble",
        "observation_models",
        "ood_detector",
        "posterior_inference_engine",
        "query_compiler",
        "reference_prior",
        "sufficiency_evaluator",
        "transition_model",
        "uncertainty_calibrator",
        "value_of_information_engine",
    }
    assert original_names <= {binding.port.value for binding in bundle.ports}

    payload = bundle.model_dump(mode="python")
    payload["ports"] = payload["ports"][:-1]
    with pytest.raises(ValidationError, match="classify every biological and support stage"):
        BiologicalModelBundleContract.model_validate(payload)


def test_port_dispositions_cannot_hide_missing_or_nonexecutable_code() -> None:
    port = BiologicalStagePort.POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL
    with pytest.raises(ValidationError, match="provided port requires"):
        ModelPortBinding(
            port=port,
            disposition=PortDisposition.PROVIDED,
            rationale=("Claimed but absent.",),
        )

    with pytest.raises(ValidationError, match="only a provided port"):
        ModelPortBinding(
            port=port,
            disposition=PortDisposition.REQUIRED,
            implementation=PortImplementationBinding(
                implementation_id="component",
                implementation_version="1",
                interface="PopulationAssayResponseModel",
                kind=PortImplementationKind.SPECIFICATION_ONLY,
                code_artifact=code_artifact(port),
            ),
            rationale=("Invalid hidden implementation.",),
        )

    specification = PortImplementationBinding(
        implementation_id="component-spec",
        implementation_version="1",
        interface="PopulationAssayResponseModel",
        kind=PortImplementationKind.SPECIFICATION_ONLY,
        code_artifact=code_artifact(port),
    )
    assert not specification.executable


@pytest.mark.parametrize("operation", tuple(ModelOperation))
def test_runtime_operations_require_their_canonical_minimum_port_set(
    operation: ModelOperation,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
) -> None:
    envelope, _ = component_contracts(query, benchmark)
    payload = envelope.model_dump(mode="python")
    payload.update(
        {
            "bundle_kind": BundleContractKind.BIOLOGICAL_MODEL_BUNDLE,
            "direct_population_response": None,
            "runtime_operations": (operation,),
            "required_ports": (BiologicalStagePort.EXACT_ARTIFACT_RESOLVER,),
        }
    )
    with pytest.raises(ValidationError, match="omits prerequisite ports"):
        BiologicalSupportEnvelope.model_validate(payload)
    assert OPERATION_REQUIRED_PORTS[operation]


def test_endpoint_component_semantics_cannot_be_recast_as_a_runtime_belief(
    query: StateQuery,
    benchmark: BenchmarkArtifact,
) -> None:
    envelope, _ = component_contracts(query, benchmark)
    payload = envelope.model_dump(mode="python")
    payload["direct_population_response"] = None
    with pytest.raises(ValidationError, match="direct component requires population-response"):
        BiologicalSupportEnvelope.model_validate(payload)

    payload = envelope.model_dump(mode="python")
    payload["runtime_operations"] = (ModelOperation.ESTIMATE_CELL_STATE,)
    with pytest.raises(ValidationError, match="cannot register"):
        BiologicalSupportEnvelope.model_validate(payload)


def test_content_drift_blocks_exact_query_binding(
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    manifest: DatasetManifest,
) -> None:
    envelope, bundle = component_contracts(query, benchmark)
    payload = bundle.model_dump(mode="python")
    query_ref = payload["query"]
    query_ref["artifact"]["sha256"] = "0" * 64
    drifted = BiologicalModelBundleContract.model_validate(payload)
    readiness = assess_biological_model_bundle(
        drifted,
        query=query,
        benchmark=benchmark,
        manifests={
            binding.binding_id: manifest for binding in benchmark.definition.evidence_bindings
        },
        support_envelope=envelope,
    )
    assert not readiness.query_binding_verified
    assert not readiness.runnable
    assert "exact query binding did not verify" in readiness.blockers


@pytest.mark.parametrize(
    ("bound_operation", "implementation_kind", "expected_bound", "expected_executable"),
    (
        (None, None, False, False),
        (
            ModelOperation.CHOOSE_INTERVENTION,
            PortImplementationKind.PYTHON_ENTRY_POINT,
            False,
            False,
        ),
        (
            ModelOperation.EVOLVE_CELL_STATE,
            PortImplementationKind.SPECIFICATION_ONLY,
            True,
            False,
        ),
    ),
)
def test_named_runtime_operation_requires_exact_executable_high_level_binding(
    bound_operation: ModelOperation | None,
    implementation_kind: PortImplementationKind | None,
    expected_bound: bool,
    expected_executable: bool,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    manifest: DatasetManifest,
) -> None:
    component_envelope, component_bundle = component_contracts(query, benchmark)
    operation = ModelOperation.EVOLVE_CELL_STATE
    required = tuple(sorted(OPERATION_REQUIRED_PORTS[operation], key=lambda port: port.value))
    envelope = BiologicalSupportEnvelope(
        envelope_id="full-evolution-envelope",
        envelope_version="0.1.0",
        bundle_kind=BundleContractKind.BIOLOGICAL_MODEL_BUNDLE,
        query=component_envelope.query,
        benchmark=component_envelope.benchmark,
        runtime_operations=(operation,),
        required_ports=required,
        required_validation_evidence=(
            ValidationEvidenceRequirement(
                evidence_id="component-validation",
                evidence_kind=ValidationEvidenceKind.RUNTIME_OPERATION_VALIDATION,
                partition_roles=(BenchmarkPartitionRole.UNTOUCHED_TEST,),
            ),
        ),
        notes=("Test-only full runtime envelope.",),
    )
    operation_bindings: tuple[ModelOperationImplementationBinding, ...] = ()
    if bound_operation is not None and implementation_kind is not None:
        operation_bindings = (
            ModelOperationImplementationBinding(
                operation=bound_operation,
                implementation=PortImplementationBinding(
                    implementation_id=f"cellstate.operation.{bound_operation.value}",
                    implementation_version="0.1.0",
                    interface=bound_operation.value,
                    kind=implementation_kind,
                    code_artifact=code_artifact(BiologicalStagePort.TRANSITION_MODEL),
                    entrypoint=(
                        f"cellstate.backends.runtime:{bound_operation.value}"
                        if implementation_kind is PortImplementationKind.PYTHON_ENTRY_POINT
                        else None
                    ),
                ),
                validation_evidence_ids=("component-validation",),
                rationale=("Test exact high-level runtime binding.",),
            ),
        )
    payload = component_bundle.model_dump(mode="python")
    payload.update(
        {
            "bundle_kind": BundleContractKind.BIOLOGICAL_MODEL_BUNDLE,
            "posterior_schema_id": "cellstate.v2.test-posterior",
            "support_envelope": contract_reference(
                envelope.envelope_id,
                envelope.envelope_version,
                envelope,
            ),
            "ports": port_map(required),
            "operation_implementations": operation_bindings,
        }
    )
    bundle = BiologicalModelBundleContract.model_validate(payload)
    readiness = assess_biological_model_bundle(
        bundle,
        query=query,
        benchmark=benchmark,
        manifests={
            binding.binding_id: manifest for binding in benchmark.definition.evidence_bindings
        },
        support_envelope=envelope,
    )
    assert readiness.runtime_operations_bound is expected_bound
    assert readiness.runtime_operations_executable is expected_executable
    assert not readiness.runtime_registration_allowed
    assert not readiness.runnable


def complete_looking_runtime_contracts(
    query: StateQuery,
    benchmark: BenchmarkArtifact,
) -> tuple[
    BiologicalSupportEnvelope,
    TrainingRunBinding,
    ValidationEvidenceBinding,
    BiologicalModelBundleContract,
]:
    """Build forgeable-looking declarations to prove the v0.1 boundary stays closed."""

    component_envelope, component_bundle = component_contracts(query, benchmark)
    operation = ModelOperation.EVOLVE_CELL_STATE
    required = tuple(sorted(OPERATION_REQUIRED_PORTS[operation], key=lambda port: port.value))
    envelope = BiologicalSupportEnvelope(
        envelope_id="synthetic-full-runtime-envelope",
        envelope_version="0.1.0",
        bundle_kind=BundleContractKind.BIOLOGICAL_MODEL_BUNDLE,
        query=component_envelope.query,
        benchmark=component_envelope.benchmark,
        runtime_operations=(operation,),
        required_ports=required,
        required_validation_evidence=(
            ValidationEvidenceRequirement(
                evidence_id="synthetic-validation",
                evidence_kind=ValidationEvidenceKind.RUNTIME_OPERATION_VALIDATION,
                partition_roles=(BenchmarkPartitionRole.UNTOUCHED_TEST,),
            ),
        ),
        notes=("Synthetic contract-path fixture; not biological evidence.",),
    )
    model_artifact = named_artifact("synthetic-model-weights")
    training = TrainingRunBinding(
        run_id="synthetic-training-run",
        run_version="0.1.0",
        query_fingerprint=query.fingerprint,
        benchmark_fingerprint=benchmark.fingerprint,
        support_envelope_fingerprint=envelope.fingerprint,
        model_artifact=model_artifact,
        training_partition_ids=("p1-train",),
        calibration_partition_ids=("p2-calibration",),
        model_selection_validation_partition_ids=("p3-model-selection-validation",),
        training_evidence_artifacts=(named_artifact("synthetic-training-evidence"),),
        calibration_evidence_artifacts=(named_artifact("synthetic-calibration-evidence"),),
        model_selection_evidence_artifacts=(named_artifact("synthetic-model-selection-evidence"),),
        model_selection_freeze_artifact=named_artifact("synthetic-model-selection-freeze"),
    )
    operation_binding = ModelOperationImplementationBinding(
        operation=operation,
        implementation=PortImplementationBinding(
            implementation_id="synthetic-evolution-operation",
            implementation_version="0.1.0",
            interface="StateEvolutionModel",
            kind=PortImplementationKind.PYTHON_ENTRY_POINT,
            code_artifact=named_artifact("synthetic-evolution-code"),
            entrypoint="tests.synthetic:evolve",
        ),
        validation_evidence_ids=("synthetic-validation",),
        rationale=("Synthetic executable surface fixture.",),
    )
    payload = component_bundle.model_dump(mode="python")
    payload.update(
        {
            "bundle_id": "synthetic-admitted-full-bundle",
            "bundle_version": "0.1.0",
            "bundle_kind": BundleContractKind.BIOLOGICAL_MODEL_BUNDLE,
            "description": "Synthetic contract-path fixture; not a biological model.",
            "posterior_schema_id": "cellstate.v2.synthetic-posterior",
            "support_envelope": contract_reference(
                envelope.envelope_id,
                envelope.envelope_version,
                envelope,
            ),
            "model_artifact": model_artifact,
            "training_run": contract_reference(training.run_id, training.run_version, training),
            "validation_evidence": (),
            "ports": port_map(
                required,
                required_disposition=PortDisposition.PROVIDED,
                evidence_id="synthetic-validation",
            ),
            "operation_implementations": (operation_binding,),
        }
    )
    declaration = BiologicalModelBundleContract.model_validate(payload)
    case_set = benchmark.definition.evaluation_case_set
    assert case_set is not None
    validation = ValidationEvidenceBinding(
        evidence_id="synthetic-validation",
        evidence_version="0.1.0",
        evidence_kind=ValidationEvidenceKind.RUNTIME_OPERATION_VALIDATION,
        query_fingerprint=query.fingerprint,
        benchmark_fingerprint=benchmark.fingerprint,
        support_envelope_fingerprint=envelope.fingerprint,
        training_run_fingerprint=training.fingerprint,
        model_artifact_fingerprint=model_artifact.sha256,
        implementation_scope_fingerprint=declaration.implementation_scope_fingerprint,
        partition_ids=("p4-untouched-test",),
        evaluation_case_ids=tuple(
            sorted(
                case.case_id for case in case_set.cases if case.partition_id == "p4-untouched-test"
            )
        ),
        covered_ports=required,
        covered_operations=(operation,),
        evidence_artifacts=(named_artifact("synthetic-validation-results"),),
    )
    final_payload = declaration.model_dump(mode="python")
    final_payload["validation_evidence"] = (
        contract_reference(
            validation.evidence_id,
            validation.evidence_version,
            validation,
        ),
    )
    bundle = BiologicalModelBundleContract.model_validate(final_payload)
    assert bundle.implementation_scope_fingerprint == declaration.implementation_scope_fingerprint
    return envelope, training, validation, bundle


def test_complete_looking_declarations_remain_nonexecutable_in_v01(
    monkeypatch: pytest.MonkeyPatch,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    manifest: DatasetManifest,
) -> None:
    envelope, training, validation, bundle = complete_looking_runtime_contracts(query, benchmark)
    manifests = {binding.binding_id: manifest for binding in benchmark.definition.evidence_bindings}
    monkeypatch.setattr(
        "cellstate.backends.contracts.verify_benchmark_artifact",
        lambda *_args, **_kwargs: SimpleNamespace(admission_ready=True),
    )
    monkeypatch.setattr(
        "cellstate.backends.contracts._benchmark_evaluation_complete",
        lambda _benchmark: True,
    )
    kwargs = {
        "query": query,
        "benchmark": benchmark,
        "manifests": manifests,
        "support_envelope": envelope,
        "training_run": training,
        "validation_evidence": (validation,),
    }
    readiness = assess_biological_model_bundle(bundle, **kwargs)
    assert readiness.query_binding_verified
    assert readiness.training_binding_verified
    assert readiness.calibration_binding_verified
    assert readiness.model_selection_binding_verified
    assert readiness.validation_bindings_verified
    assert readiness.validation_evidence_semantics_verified
    assert readiness.implementation_scope_binding_verified
    assert readiness.required_ports_provided
    assert not readiness.required_ports_evidenced
    assert readiness.runtime_operations_bound
    assert not readiness.runtime_operations_evidenced
    assert readiness.benchmark_admission_ready
    assert not readiness.artifact_bytes_resolved
    assert not readiness.implementation_interfaces_verified
    assert not readiness.validation_results_verified
    assert not readiness.validation_results_passed
    assert not readiness.query_derived_prerequisites_verified
    assert not readiness.required_ports_executable
    assert not readiness.runtime_operations_executable
    assert not readiness.component_evaluation_complete
    assert readiness.lifecycle_stage is ComponentLifecycleStage.SCAFFOLD
    assert not readiness.component_executable
    assert not readiness.scientifically_admitted
    assert not readiness.runtime_registration_allowed
    assert not readiness.runnable
    assert set(readiness.admission_blocker_codes) == {
        BundleAdmissionBlockerCode.ARTIFACT_BYTES_UNRESOLVED,
        BundleAdmissionBlockerCode.IMPLEMENTATION_INTERFACES_UNVERIFIED,
        BundleAdmissionBlockerCode.QUERY_DERIVED_OPERATION_PREREQUISITES_UNVERIFIED,
        BundleAdmissionBlockerCode.VALIDATION_RESULTS_UNVERIFIED,
    }
    with pytest.raises(BiologicalExecutionBlockedError, match="trusted"):
        require_biological_execution(
            bundle,
            operation=ModelOperation.EVOLVE_CELL_STATE,
            **kwargs,
        )
    with pytest.raises(BiologicalExecutionBlockedError, match="trusted"):
        require_biological_execution(
            bundle,
            operation=ModelOperation.ESTIMATE_CELL_STATE,
            **kwargs,
        )
    with pytest.raises(BiologicalExecutionBlockedError, match="trusted"):
        build_admitted_estimator_descriptor(bundle, **kwargs)


def test_model_selected_lifecycle_can_precede_semantic_validation_results(
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    manifest: DatasetManifest,
) -> None:
    envelope, bundle = component_contracts(query, benchmark)
    readiness = assess_biological_model_bundle(
        bundle,
        query=query,
        benchmark=benchmark,
        manifests={
            binding.binding_id: manifest for binding in benchmark.definition.evidence_bindings
        },
        support_envelope=envelope,
    )
    payload = readiness.model_dump(mode="python")
    payload.update(
        {
            "artifact_bytes_resolved": True,
            "implementation_interfaces_verified": True,
            "query_derived_prerequisites_verified": True,
            "training_binding_verified": True,
            "calibration_binding_verified": True,
            "model_selection_binding_verified": True,
            "admission_blocker_codes": (BundleAdmissionBlockerCode.VALIDATION_RESULTS_UNVERIFIED,),
            "lifecycle_stage": ComponentLifecycleStage.MODEL_SELECTED_FROZEN,
        }
    )

    candidate = type(readiness).model_validate(payload)
    assert candidate.lifecycle_stage is ComponentLifecycleStage.MODEL_SELECTED_FROZEN
    assert not candidate.validation_results_verified


@pytest.mark.parametrize("drift", ("port_code", "posterior_schema"))
def test_validation_scope_fingerprint_detects_implementation_or_schema_drift(
    drift: str,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    manifest: DatasetManifest,
) -> None:
    envelope, training, validation, bundle = complete_looking_runtime_contracts(query, benchmark)
    payload = bundle.model_dump(mode="python")
    if drift == "port_code":
        provided = next(
            binding
            for binding in payload["ports"]
            if binding["disposition"] == PortDisposition.PROVIDED
        )
        provided["implementation"]["code_artifact"]["sha256"] = "0" * 64
    else:
        payload["posterior_schema_id"] = "cellstate.v2.drifted-posterior"
    drifted = BiologicalModelBundleContract.model_validate(payload)
    readiness = assess_biological_model_bundle(
        drifted,
        query=query,
        benchmark=benchmark,
        manifests={
            binding.binding_id: manifest for binding in benchmark.definition.evidence_bindings
        },
        support_envelope=envelope,
        training_run=training,
        validation_evidence=(validation,),
    )
    assert readiness.validation_bindings_verified
    assert not readiness.implementation_scope_binding_verified
    assert not readiness.required_ports_evidenced
    assert not readiness.runnable


@pytest.mark.parametrize("drift", ("kind", "partition"))
def test_typed_validation_role_cannot_be_replaced_by_a_generic_result_artifact(
    drift: str,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    manifest: DatasetManifest,
) -> None:
    envelope, training, validation, bundle = complete_looking_runtime_contracts(query, benchmark)
    evidence_payload = validation.model_dump(mode="python")
    if drift == "kind":
        evidence_payload["evidence_kind"] = ValidationEvidenceKind.MODEL_SELECTION
    else:
        case_set = benchmark.definition.evaluation_case_set
        assert case_set is not None
        evidence_payload["partition_ids"] = ("p3-model-selection-validation",)
        evidence_payload["evaluation_case_ids"] = tuple(
            sorted(
                case.case_id
                for case in case_set.cases
                if case.partition_id == "p3-model-selection-validation"
            )
        )
    drifted_evidence = ValidationEvidenceBinding.model_validate(evidence_payload)
    bundle_payload = bundle.model_dump(mode="python")
    bundle_payload["validation_evidence"] = (
        contract_reference(
            drifted_evidence.evidence_id,
            drifted_evidence.evidence_version,
            drifted_evidence,
        ),
    )
    drifted_bundle = BiologicalModelBundleContract.model_validate(bundle_payload)
    readiness = assess_biological_model_bundle(
        drifted_bundle,
        query=query,
        benchmark=benchmark,
        manifests={
            binding.binding_id: manifest for binding in benchmark.definition.evidence_bindings
        },
        support_envelope=envelope,
        training_run=training,
        validation_evidence=(drifted_evidence,),
    )
    assert readiness.validation_bindings_verified
    assert not readiness.validation_evidence_semantics_verified
    assert not readiness.required_ports_evidenced
    assert not readiness.runnable
