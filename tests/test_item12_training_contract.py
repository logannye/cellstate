"""Trusted p1-only transition from a component scaffold to a trained candidate."""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

import cellstate.backends.training as training_contracts
from cellstate.backends import (
    TRAINED_CANDIDATE_FACTORY_INTERFACE,
    AdmissionArtifactKind,
    AdmissionVerifierCapability,
    AdmissionVerifierIdentity,
    BiologicalModelBundleContract,
    BiologicalStagePort,
    BiologicalSupportEnvelope,
    BundleContractKind,
    BundleContractReference,
    CandidateTrainingPlan,
    ComponentLifecycleStage,
    ImplementationReceiptTargetKind,
    InterfaceConformanceObservation,
    InterfaceVerificationMethod,
    LoadedObjectIdentity,
    LoadedObjectKind,
    ModelPortBinding,
    P1TrainingEvidence,
    PortDisposition,
    PortImplementationBinding,
    PortImplementationKind,
    TrainedCandidateVerification,
    TrainingRunBinding,
    TrainingVerificationContext,
    TrustedAdmissionVerifier,
    TrustedRuntimeInterface,
    assess_biological_model_bundle,
    attest_isolated_loaded_interface_observation,
    derive_query_prerequisite_report,
    implementation_requirement_for_binding,
    issue_artifact_resolution_receipt,
    issue_candidate_fit_receipt,
    issue_loaded_interface_receipt,
    issue_training_source_selection_receipt,
    require_exact_trained_candidate,
    trained_candidate_required_artifacts,
    training_evidence_artifacts_for_context,
)
from cellstate.data import (
    BenchmarkArtifact,
    BenchmarkPartitionRole,
    ContentAddressedArtifact,
    DatasetManifest,
    SourceArtifact,
    SourceKind,
)
from cellstate.domain import StateQuery
from cellstate.domain.common import SchemaModel, canonical_fingerprint, canonical_json_bytes
from cellstate.training.execution import (
    ContainedExecutionObservation,
    ContainedExecutionPolicy,
    ContainedTrainingObservation,
    ContainedTrainingWorkerObservation,
    ExecutionInputClosureManifest,
    RuntimeBuilderIdentity,
    RuntimeImageIdentity,
    RuntimeImageLayerIdentity,
    RuntimeImageLock,
    StagedTrainingEntry,
    StagedTrainingInventory,
    TrainingCodeClosureEntry,
    TrainingCodeClosureManifest,
)
from cellstate.training.publication import generation_id_for_seed

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/benchmark-artifact.json"
BUNDLE_PATH = ROOT / "backends/vertical-a/sciplex3-k562-24h-v1/bundle-contract.json"
MANIFEST_PATH = ROOT / "data_manifests/reviewed/sciplex3-k562-24h.json"
QUERY_PATH = ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/state-query.json"
SUPPORT_PATH = ROOT / "backends/vertical-a/sciplex3-k562-24h-v1/support-envelope.json"
NOW = datetime(2026, 8, 10, 18, 0, tzinfo=UTC)
MODEL_BYTES = b'{"candidate":"sealed-p1-model"}'
SOURCE_BYTES = b"exact-p1-source"


class CandidateImplementation:
    """Small exact class used as the application-owned candidate interface registry entry."""

    @classmethod
    def load_exact(cls, model_bytes: bytes, *, expected_sha256: str) -> object:
        if sha256(model_bytes).hexdigest() != expected_sha256:
            raise ValueError("wrong model bytes")
        return cls()

    def supports(self, target: object) -> bool:
        return target is not None

    def sample(self, request: object) -> object:
        return request

    @property
    def model_artifact_sha256(self) -> str:
        return sha256(MODEL_BYTES).hexdigest()

    def model_bytes(self) -> bytes:
        return MODEL_BYTES

    def behavior_manifest(self) -> Mapping[str, object]:
        return {"heldout_read": False}


class CandidateImplementationWithoutArtifactHash:
    """Adversarial registry entry matching the old, incomplete verifier minimum."""

    @classmethod
    def load_exact(cls, model_bytes: bytes, *, expected_sha256: str) -> object:
        if sha256(model_bytes).hexdigest() != expected_sha256:
            raise ValueError("wrong model bytes")
        return cls()

    def supports(self, target: object) -> bool:
        return target is not None

    def sample(self, request: object) -> object:
        return request

    def model_bytes(self) -> bytes:
        return MODEL_BYTES

    def behavior_manifest(self) -> Mapping[str, object]:
        return {"heldout_read": False}


@dataclass(frozen=True, slots=True)
class TrainingFixture:
    query: StateQuery
    benchmark: BenchmarkArtifact
    manifest: DatasetManifest
    support: BiologicalSupportEnvelope
    bundle: BiologicalModelBundleContract
    training_run: TrainingRunBinding
    context: TrainingVerificationContext
    content_by_sha256: Mapping[str, bytes]


def _artifact(
    artifact_id: str,
    content: bytes,
    content_by_sha256: dict[str, bytes],
    *,
    media_type: str = "application/octet-stream",
) -> ContentAddressedArtifact:
    digest = sha256(content).hexdigest()
    content_by_sha256[digest] = content
    return ContentAddressedArtifact(
        artifact_id=artifact_id,
        uri=f"https://example.invalid/item12/{artifact_id}",
        sha256=digest,
        byte_count=len(content),
        media_type=media_type,
    )


def _contract_artifact(
    artifact_id: str,
    model: SchemaModel,
    content_by_sha256: dict[str, bytes],
) -> ContentAddressedArtifact:
    content = canonical_json_bytes(model.model_dump(mode="json"))
    return _artifact(
        artifact_id,
        content,
        content_by_sha256,
        media_type="application/json",
    )


def _contract_reference(
    contract_id: str,
    contract_version: str,
    model: SchemaModel,
    content_by_sha256: dict[str, bytes],
) -> BundleContractReference:
    return BundleContractReference(
        contract_id=contract_id,
        contract_version=contract_version,
        artifact=_contract_artifact(
            f"{contract_id}-canonical-json",
            model,
            content_by_sha256,
        ),
    )


def _trusted_verifier(content_by_sha256: dict[str, bytes]) -> TrustedAdmissionVerifier:
    capabilities = tuple(sorted(AdmissionVerifierCapability, key=lambda item: item.value))
    identity = AdmissionVerifierIdentity(
        verifier_id="cellstate.item12-training-verifier",
        verifier_version="0.1.0",
        code_artifact=_artifact(
            "item12-training-verifier-code",
            b"item12-verifier-code",
            content_by_sha256,
            media_type="text/x-python",
        ),
        entrypoint="cellstate.backends.training:item12_verifier",
        runtime="cpython-3.11",
        capabilities=capabilities,
    )
    return TrustedAdmissionVerifier(
        identity=identity,
        key_id="item12-test-key-v1",
        secret=b"t" * 32,
    )


def _training_fixture(
    *,
    training_partition_ids: tuple[str, ...] = ("p1-train",),
    candidate_factory_class: type[object] = CandidateImplementation,
) -> TrainingFixture:
    content_by_sha256: dict[str, bytes] = {}
    query = StateQuery.model_validate_json(QUERY_PATH.read_bytes())
    benchmark = BenchmarkArtifact.model_validate_json(BENCHMARK_PATH.read_bytes())
    manifest = DatasetManifest.model_validate_json(MANIFEST_PATH.read_bytes())
    base_support = BiologicalSupportEnvelope.model_validate_json(SUPPORT_PATH.read_bytes())
    support_payload = base_support.model_dump(mode="python")
    support_payload["bundle_kind"] = BundleContractKind.COMPONENT_MODEL
    support = BiologicalSupportEnvelope.model_validate(support_payload)

    query_reference = support.query
    benchmark_reference = support.benchmark
    content_by_sha256[query_reference.artifact.sha256] = canonical_json_bytes(
        query.model_dump(mode="json")
    )
    content_by_sha256[benchmark_reference.artifact.sha256] = canonical_json_bytes(
        benchmark.model_dump(mode="json")
    )
    support_reference = _contract_reference(
        support.envelope_id,
        support.envelope_version,
        support,
        content_by_sha256,
    )

    trainer_code = _artifact(
        "item12-candidate-trainer-code",
        b"candidate-trainer-code",
        content_by_sha256,
        media_type="text/x-python",
    )
    candidate_code = _artifact(
        "item12-candidate-factory-code",
        b"candidate-factory-code",
        content_by_sha256,
        media_type="text/x-python",
    )
    trainer = PortImplementationBinding(
        implementation_id="cellstate.item12.candidate-trainer",
        implementation_version="0.1.0",
        interface="cellstate.backends.CandidateTrainer",
        kind=PortImplementationKind.PYTHON_ENTRY_POINT,
        code_artifact=trainer_code,
        entrypoint="cellstate.evaluation.candidate_runner:fit_candidate",
    )
    candidate_factory = PortImplementationBinding(
        implementation_id="cellstate.item12.candidate-factory",
        implementation_version="0.1.0",
        interface=TRAINED_CANDIDATE_FACTORY_INTERFACE,
        kind=PortImplementationKind.PYTHON_ENTRY_POINT,
        code_artifact=candidate_code,
        entrypoint=f"{__name__}:{candidate_factory_class.__name__}",
    )
    count_stream_sha256 = sha256(b"conceptual-p1-count-stream").hexdigest()
    scan_fingerprint = sha256(b"finalized-p1-scan").hexdigest()
    assembly_fingerprint = sha256(b"p1-assembly").hexdigest()
    closure_entries = tuple(
        sorted(
            (
                TrainingCodeClosureEntry(
                    relative_path="candidate.py",
                    sha256=candidate_code.sha256,
                    byte_count=candidate_code.byte_count,
                ),
                TrainingCodeClosureEntry(
                    relative_path="trainer.py",
                    sha256=trainer_code.sha256,
                    byte_count=trainer_code.byte_count,
                ),
            ),
            key=lambda item: item.relative_path,
        )
    )
    code_closure = TrainingCodeClosureManifest(entries=closure_entries)
    input_closure = ExecutionInputClosureManifest(
        training_code_closure_sha256=code_closure.fingerprint,
        entries=closure_entries,
    )
    image = RuntimeImageIdentity(
        reference="example.invalid/cellstate@sha256:" + "d" * 64,
        digest="sha256:" + "d" * 64,
    )
    policy = ContainedExecutionPolicy(
        policy_id="item12-test-fit",
        owner_id="item12-test",
        runtime_image=image,
        training_code_closure_sha256=code_closure.fingerprint,
        execution_input_closure_sha256=input_closure.fingerprint,
        wall_clock_seconds=60,
        cleanup_timeout_seconds=5,
        memory_max_bytes=4 * 1024**3,
        memory_swap_max_bytes=4 * 1024**3,
        pids_limit=16,
        temporary_max_bytes=1024**2,
        snapshot_max_bytes=1024**2,
        observed_training_peak_memory_bytes=1024,
        source_container_path="/run/source/source.h5ad",
        code_container_path="/workspace",
        output_container_path="/run/output",
        snapshot_container_path="/run/snapshot",
        temporary_container_path="/run/tmp",
        workdir="/workspace",
        environment={"TMPDIR": "/run/tmp"},
        worker_command=("worker.py",),
    )
    image_lock = RuntimeImageLock(
        runtime_image=image,
        builder=RuntimeBuilderIdentity(
            buildx_version="v0.28.0",
            buildx_commit="1" * 40,
            buildkit_version="v0.24.0",
            buildkit_image_digest="sha256:" + "2" * 64,
            dockerfile_frontend_digest="sha256:" + "3" * 64,
            dockerfile_sha256="4" * 64,
            requirements_sha256="5" * 64,
            source_date_epoch=1_786_406_400,
            no_cache=True,
            provenance_attestation_disabled=True,
            image_tag="example.invalid/cellstate:locked",
            output_options=("type=oci",),
        ),
        archive_sha256="6" * 64,
        oci_index_digest="sha256:" + "7" * 64,
        config_digest="sha256:" + "8" * 64,
        layers=(
            RuntimeImageLayerIdentity(
                digest="sha256:" + "9" * 64,
                byte_count=1,
            ),
        ),
        training_code_closure_sha256=code_closure.fingerprint,
        image_provenance_sha256="e" * 64,
    )
    plan_fields: dict[str, object] = {
        "schema_version": "0.1-experimental",
        "plan_id": "sciplex3-k562-item12-training-plan",
        "plan_version": "0.1.0",
        "query_fingerprint": query.fingerprint,
        "benchmark_fingerprint": benchmark.fingerprint,
        "support_envelope_fingerprint": support.fingerprint,
        "training_partition_ids": training_partition_ids,
        "training_partition_roles": (BenchmarkPartitionRole.TRAIN,),
        "p1_loader_contract": _artifact(
            "item12-p1-loader-contract",
            b"p1-loader-contract",
            content_by_sha256,
            media_type="application/json",
        ),
        "p1_count_stream": _artifact(
            "item12-p1-count-stream-descriptor",
            canonical_json_bytes({"count_stream_sha256": count_stream_sha256}),
            content_by_sha256,
            media_type="application/json",
        ),
        "p1_count_stream_sha256": count_stream_sha256,
        "p1_finalized_count_scan_fingerprint": scan_fingerprint,
        "p1_assembly_fingerprint": assembly_fingerprint,
        "p1_design_fingerprint": sha256(b"p1-design").hexdigest(),
        "ordered_feature_keys_sha256": sha256(b"ordered-features").hexdigest(),
        "action_binding_sha256": sha256(b"action-binding").hexdigest(),
        "target_value_schema_sha256": sha256(b"target-schema").hexdigest(),
        "candidate_specification": _artifact(
            "item12-candidate-specification",
            b"candidate-specification",
            content_by_sha256,
            media_type="application/json",
        ),
        "output_model_schema": _artifact(
            "item12-candidate-model-schema",
            b"candidate-model-schema",
            content_by_sha256,
            media_type="application/schema+json",
        ),
        "runtime_lock": _artifact(
            "item12-candidate-runtime-lock",
            b"candidate-runtime-lock",
            content_by_sha256,
            media_type="application/json",
        ),
        "contained_execution_policy": _contract_artifact(
            "item12-contained-execution-policy", policy, content_by_sha256
        ),
        "runtime_image_lock": _contract_artifact(
            "item12-runtime-image-lock", image_lock, content_by_sha256
        ),
        "training_code_closure": _contract_artifact(
            "item12-training-code-closure", code_closure, content_by_sha256
        ),
        "training_execution_input_closure": _contract_artifact(
            "item12-training-execution-input-closure", input_closure, content_by_sha256
        ),
        "trainer_implementation": trainer,
        "candidate_factory_implementation": candidate_factory,
        "optimization_seed": 17,
        "deterministic_thread_count": 1,
        "future_calibration_plan": None,
    }
    generation_seed = training_contracts.candidate_training_plan_generation_seed_bytes(plan_fields)
    planned_generation_id = generation_id_for_seed(generation_seed)
    plan = CandidateTrainingPlan(
        **plan_fields,
        planned_generation_id=planned_generation_id,
        publication_generation_seed=_artifact(
            "item12-publication-generation-seed",
            generation_seed,
            content_by_sha256,
            media_type="application/json",
        ),
    )
    plan_artifact = _contract_artifact("item12-training-plan", plan, content_by_sha256)

    source = SourceArtifact(
        source_id="item12-exact-p1-source",
        kind=SourceKind.PROCESSED,
        uri="https://example.invalid/item12/exact-p1-source.h5ad",
        sha256=sha256(SOURCE_BYTES).hexdigest(),
        media_type="application/x-hdf5",
        accession="GSE139944",
        release="2020-04-30",
        parent_study_accession="GSE139944",
        parent_study_release="2020-04-30",
        byte_count=len(SOURCE_BYTES),
        retrieved_at=NOW,
    )
    content_by_sha256[source.sha256] = SOURCE_BYTES
    workflow = _artifact(
        "item12-p1-source-workflow-resolution",
        b"typed-p1-source-workflow-resolution",
        content_by_sha256,
        media_type="application/json",
    )
    trusted_verifier = _trusted_verifier(content_by_sha256)
    source_selection = issue_training_source_selection_receipt(
        selection_id="item12-p1-source-selection",
        plan=plan,
        workflow_resolution_artifacts=(workflow,),
        sources=(source,),
        trusted_selector=trusted_verifier,
        issued_at=NOW,
    )

    finalized_count_scan = _artifact(
        "item12-p1-finalized-count-scan",
        b"p1-finalized-count-scan",
        content_by_sha256,
        media_type="application/json",
    )
    assembly_receipt = _artifact(
        "item12-p1-assembly-receipt",
        b"p1-assembly-receipt",
        content_by_sha256,
        media_type="application/json",
    )
    p1_materialization = _artifact(
        "item12-p1-materialization",
        b"p1-materialization",
        content_by_sha256,
        media_type="application/json",
    )
    training_result = _artifact(
        "item12-candidate-training-result",
        b"candidate-training-result-observation",
        content_by_sha256,
        media_type="application/json",
    )
    model_artifact = _artifact(
        "item12-p1-trained-candidate-model",
        MODEL_BYTES,
        content_by_sha256,
        media_type="application/vnd.cellstate.candidate+json",
    )
    staged_inventory = StagedTrainingInventory(
        entries=tuple(
            sorted(
                (
                    StagedTrainingEntry(
                        relative_path="candidate-model.json",
                        artifact_role="model_artifact",
                        sha256=model_artifact.sha256,
                        byte_count=model_artifact.byte_count,
                    ),
                    StagedTrainingEntry(
                        relative_path="candidate-training-plan.json",
                        artifact_role="training_plan",
                        sha256=plan_artifact.sha256,
                        byte_count=plan_artifact.byte_count,
                    ),
                    StagedTrainingEntry(
                        relative_path="p1-assembly-receipt.json",
                        artifact_role="p1_assembly_receipt",
                        sha256=assembly_receipt.sha256,
                        byte_count=assembly_receipt.byte_count,
                    ),
                    StagedTrainingEntry(
                        relative_path="p1-finalized-count-scan-receipt.json",
                        artifact_role="p1_finalized_count_scan",
                        sha256=finalized_count_scan.sha256,
                        byte_count=finalized_count_scan.byte_count,
                    ),
                    StagedTrainingEntry(
                        relative_path="training-execution-observation.json",
                        artifact_role="training_result",
                        sha256=training_result.sha256,
                        byte_count=training_result.byte_count,
                    ),
                ),
                key=lambda item: item.relative_path,
            )
        )
    )
    parent_execution = ContainedExecutionObservation(
        execution_id="item12-fit",
        policy_fingerprint=policy.fingerprint,
        runtime_image_digest=image.digest,
        container_user_mode=policy.container_user_mode,
        observed_container_uid=1000,
        observed_container_gid=1000,
        outcome="success",
        exit_code=0,
        timed_out=False,
        oom_killed=False,
    )
    worker_execution = ContainedTrainingWorkerObservation(
        execution_id="item12-fit",
        training_plan_fingerprint=plan.fingerprint,
        policy_fingerprint=policy.fingerprint,
        runtime_image_digest=image.digest,
        training_code_closure_sha256=code_closure.fingerprint,
        execution_input_closure_sha256=input_closure.fingerprint,
        expected_source_sha256=source.sha256,
        source_pre_sha256=source.sha256,
        source_post_sha256=source.sha256,
        expected_source_byte_count=source.byte_count,
        source_pre_byte_count=source.byte_count,
        source_post_byte_count=source.byte_count,
        staged_inventory=staged_inventory,
        staged_tree_sha256=staged_inventory.fingerprint,
        staged_file_count=len(staged_inventory.entries),
    )
    contained_execution = ContainedTrainingObservation(
        training_plan_fingerprint=plan.fingerprint,
        policy_fingerprint=policy.fingerprint,
        runtime_image_digest=image.digest,
        training_code_closure_sha256=code_closure.fingerprint,
        execution_input_closure_sha256=input_closure.fingerprint,
        staged_inventory=staged_inventory,
        staged_tree_sha256=staged_inventory.fingerprint,
        worker_observation=worker_execution,
        execution_observation=parent_execution,
    )
    contained_execution_artifact = _contract_artifact(
        "item12-contained-execution-observation",
        contained_execution,
        content_by_sha256,
    )
    p1_evidence = P1TrainingEvidence(
        evidence_id="item12-real-p1-training-evidence",
        evidence_version="0.1.0",
        training_plan_fingerprint=plan.fingerprint,
        partition_ids=training_partition_ids,
        source=source,
        finalized_count_scan=finalized_count_scan,
        assembly_receipt=assembly_receipt,
        p1_materialization=p1_materialization,
        contained_execution_observation=contained_execution_artifact,
        count_stream_sha256=count_stream_sha256,
        finalized_count_scan_fingerprint=scan_fingerprint,
        assembly_fingerprint=assembly_fingerprint,
        record_count=94_785,
        well_count=768,
        treated_well_count=752,
        control_well_count=16,
        nnz=10_000,
        zero_panel_record_count=7,
    )
    p1_evidence_artifact = _contract_artifact(
        "item12-p1-training-evidence",
        p1_evidence,
        content_by_sha256,
    )
    fit_receipt = issue_candidate_fit_receipt(
        receipt_id="item12-candidate-fit-receipt",
        plan=plan,
        source_selection=source_selection,
        p1_evidence=p1_evidence,
        contained_execution_observation=contained_execution,
        training_result=training_result,
        model_artifact=model_artifact,
        observed_model_content=MODEL_BYTES,
        behavior_manifest_sha256=canonical_fingerprint({"heldout_read": False}),
        trusted_verifier=trusted_verifier,
        issued_at=NOW,
    )

    deterministic_training_evidence = tuple(
        sorted(
            {
                artifact.artifact_id: artifact
                for artifact in (
                    plan_artifact,
                    p1_evidence_artifact,
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
                    p1_evidence.finalized_count_scan,
                    p1_evidence.assembly_receipt,
                    p1_evidence.p1_materialization,
                    p1_evidence.contained_execution_observation,
                    fit_receipt.training_result,
                    *source_selection.workflow_resolution_artifacts,
                )
            }.values(),
            key=lambda item: item.artifact_id,
        )
    )
    training_run = TrainingRunBinding(
        run_id="item12-p1-candidate-training-run",
        run_version="0.1.0",
        query_fingerprint=query.fingerprint,
        benchmark_fingerprint=benchmark.fingerprint,
        support_envelope_fingerprint=support.fingerprint,
        model_artifact=model_artifact,
        training_partition_ids=training_partition_ids,
        training_evidence_artifacts=deterministic_training_evidence,
    )
    training_run_reference = _contract_reference(
        training_run.run_id,
        training_run.run_version,
        training_run,
        content_by_sha256,
    )

    base_bundle = BiologicalModelBundleContract.model_validate_json(BUNDLE_PATH.read_bytes())
    ports: list[ModelPortBinding] = []
    for binding in base_bundle.ports:
        if binding.port is BiologicalStagePort.POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL:
            ports.append(
                ModelPortBinding(
                    port=binding.port,
                    disposition=PortDisposition.PROVIDED,
                    implementation=candidate_factory,
                    rationale=("Exact p1-trained candidate factory; not a public runtime.",),
                )
            )
        else:
            ports.append(binding)
    bundle = BiologicalModelBundleContract(
        bundle_id="sciplex3-k562-p1-trained-candidate",
        bundle_version="0.1.0",
        bundle_kind=BundleContractKind.COMPONENT_MODEL,
        description="P1-trained candidate with calibration and all public runtime closed.",
        query=query_reference,
        benchmark=benchmark_reference,
        support_envelope=support_reference,
        model_artifact=model_artifact,
        training_run=training_run_reference,
        ports=tuple(sorted(ports, key=lambda item: item.port.value)),
    )
    prerequisite_report = derive_query_prerequisite_report(
        query=query,
        support_envelope=support,
        bundle=bundle,
    )
    assert prerequisite_report.structurally_satisfied

    interface_artifact = _artifact(
        "item12-candidate-factory-interface",
        b"candidate-factory-interface",
        content_by_sha256,
        media_type="application/schema+json",
    )
    trusted_interface = TrustedRuntimeInterface(
        declared_interface=TRAINED_CANDIDATE_FACTORY_INTERFACE,
        interface_artifact=interface_artifact,
        runtime_interface=candidate_factory_class,
    )
    requirement = implementation_requirement_for_binding(
        bundle,
        target_kind=ImplementationReceiptTargetKind.PORT,
        target_id=BiologicalStagePort.POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL.value,
        implementation=candidate_factory,
        trusted_runtime_interface=trusted_interface,
    )
    loaded_identity = LoadedObjectIdentity(
        entrypoint=candidate_factory.entrypoint or "",
        module=candidate_factory_class.__module__,
        qualname=candidate_factory_class.__qualname__,
        object_kind=LoadedObjectKind.CLASS,
        loaded_code_sha256=candidate_code.sha256,
    )
    conformance = InterfaceConformanceObservation(
        verification_method=InterfaceVerificationMethod.INSPECTED_SIGNATURE_SET,
        declared_interface=TRAINED_CANDIDATE_FACTORY_INTERFACE,
        runtime_interface_module=requirement.runtime_interface_module,
        runtime_interface_qualname=requirement.runtime_interface_qualname,
        loaded_object_fingerprint=loaded_identity.fingerprint,
        required_member_signatures=requirement.required_member_signatures,
        observed_matching_member_signatures=requirement.required_member_signatures,
        interface_contract_fingerprint=requirement.interface_contract_fingerprint,
        observed_contract_fingerprint=requirement.interface_contract_fingerprint,
    )
    isolated = attest_isolated_loaded_interface_observation(
        observation_id="item12-candidate-factory-isolated-load",
        requirement=requirement,
        loaded_object=loaded_identity,
        conformance=conformance,
        trusted_loader=trusted_verifier,
        issued_at=NOW,
        isolation_evidence_artifact=_artifact(
            "item12-candidate-factory-isolation-evidence",
            b"candidate-factory-isolation-evidence",
            content_by_sha256,
        ),
    )
    interface_receipt = issue_loaded_interface_receipt(
        receipt_id="item12-candidate-factory-interface-receipt",
        requirement=requirement,
        isolated_observation=isolated,
        trusted_runtime_interface=trusted_interface,
        trusted_loader=trusted_verifier,
        trusted_verifier=trusted_verifier,
        issued_at=NOW,
        evidence_artifacts=(
            _artifact(
                "item12-candidate-interface-audit",
                b"candidate-interface-audit",
                content_by_sha256,
            ),
        ),
    )
    partial_context = TrainingVerificationContext(
        plan=plan,
        plan_artifact=plan_artifact,
        p1_evidence=p1_evidence,
        p1_evidence_artifact=p1_evidence_artifact,
        contained_execution_policy=policy,
        runtime_image_lock=image_lock,
        training_code_closure=code_closure,
        training_execution_input_closure=input_closure,
        contained_execution_observation=contained_execution,
        source_selection=source_selection,
        fit_receipt=fit_receipt,
        query_prerequisite_report=prerequisite_report,
        artifact_receipts=(),
        candidate_factory_interface_receipt=interface_receipt,
        trusted_verifiers=(trusted_verifier,),
        runtime_interfaces={TRAINED_CANDIDATE_FACTORY_INTERFACE: trusted_interface},
    )
    assert training_evidence_artifacts_for_context(partial_context) == (
        deterministic_training_evidence
    )
    required_artifacts = trained_candidate_required_artifacts(
        bundle,
        context=partial_context,
        candidate_interface_requirement=requirement,
    )
    receipts = []
    for index, reference in enumerate(required_artifacts):
        declaration = (
            reference.content_addressed_artifact
            if reference.content_addressed_artifact is not None
            else reference.dataset_source_artifact
        )
        assert declaration is not None
        receipts.append(
            issue_artifact_resolution_receipt(
                receipt_id=f"item12-stage-artifact-{index:02d}",
                artifact=declaration,
                observed_content=content_by_sha256[reference.sha256],
                trusted_verifier=trusted_verifier,
                issued_at=NOW,
                evidence_artifacts=(
                    _artifact(
                        f"item12-stage-artifact-audit-{index:02d}",
                        f"artifact-audit-{index:02d}".encode(),
                        content_by_sha256,
                    ),
                ),
            )
        )
    context = replace(partial_context, artifact_receipts=tuple(receipts))
    return TrainingFixture(
        query=query,
        benchmark=benchmark,
        manifest=manifest,
        support=support,
        bundle=bundle,
        training_run=training_run,
        context=context,
        content_by_sha256=content_by_sha256,
    )


@pytest.fixture(scope="module")
def training_fixture() -> TrainingFixture:
    return _training_fixture()


def test_exact_p1_context_derives_only_trained_candidate(
    training_fixture: TrainingFixture,
) -> None:
    fixture = training_fixture
    verification = require_exact_trained_candidate(
        fixture.bundle,
        query=fixture.query,
        benchmark=fixture.benchmark,
        support_envelope=fixture.support,
        training_run=fixture.training_run,
        context=fixture.context,
    )
    assert type(verification) is TrainedCandidateVerification
    assert verification.model_artifact_sha256 == fixture.training_run.model_artifact.sha256
    factory_requirement = fixture.context.candidate_factory_interface_receipt.requirement
    assert any(
        signature.startswith("model_artifact_sha256|property|")
        for signature in factory_requirement.required_member_signatures
    )
    assert fixture.context.plan.training_partition_roles == (BenchmarkPartitionRole.TRAIN,)

    readiness = assess_biological_model_bundle(
        fixture.bundle,
        query=fixture.query,
        benchmark=fixture.benchmark,
        manifests={
            binding.binding_id: fixture.manifest
            for binding in fixture.benchmark.definition.evidence_bindings
        },
        support_envelope=fixture.support,
        training_run=fixture.training_run,
        training_context=fixture.context,
    )
    assert readiness.training_binding_verified
    assert readiness.training_artifacts_verified
    assert readiness.training_interfaces_verified
    assert readiness.training_result_semantics_verified
    assert readiness.trained_candidate_verified
    assert readiness.trained_candidate_verification_fingerprint == (
        verification.verification_fingerprint
    )
    assert readiness.lifecycle_stage is ComponentLifecycleStage.TRAINED_CANDIDATE
    assert not readiness.calibration_binding_verified
    assert not readiness.calibration_result_semantics_verified
    assert not readiness.model_selection_binding_verified
    assert not readiness.model_selection_result_semantics_verified
    assert not readiness.artifact_bytes_resolved
    assert not readiness.implementation_interfaces_verified
    assert not readiness.component_execution_allowed
    assert not readiness.runtime_registration_allowed
    assert not readiness.runnable

    ports = {binding.port: binding for binding in fixture.bundle.ports}
    assert (
        ports[BiologicalStagePort.POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL].disposition
        is PortDisposition.PROVIDED
    )
    assert all(
        ports[port].disposition is PortDisposition.REQUIRED
        for port in fixture.support.required_ports
        if port is not BiologicalStagePort.POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL
    )


def test_trained_candidate_factory_requires_artifact_identity_property() -> None:
    fixture = _training_fixture(
        candidate_factory_class=CandidateImplementationWithoutArtifactHash,
    )
    with pytest.raises(ValueError, match="artifact-state members"):
        require_exact_trained_candidate(
            fixture.bundle,
            query=fixture.query,
            benchmark=fixture.benchmark,
            support_envelope=fixture.support,
            training_run=fixture.training_run,
            context=fixture.context,
        )


def test_consistently_renamed_training_partition_cannot_bypass_benchmark_role_binding() -> None:
    fixture = _training_fixture(training_partition_ids=("renamed-p1-train",))
    assert (
        fixture.context.plan.training_partition_ids
        == fixture.context.p1_evidence.partition_ids
        == fixture.training_run.training_partition_ids
        == ("renamed-p1-train",)
    )
    assert fixture.benchmark.definition.split_plan is not None
    assert tuple(
        partition.partition_id
        for partition in fixture.benchmark.definition.split_plan.partitions
        if partition.role is BenchmarkPartitionRole.TRAIN
    ) == ("p1-train",)

    with pytest.raises(ValueError, match="benchmark's exact training partitions"):
        require_exact_trained_candidate(
            fixture.bundle,
            query=fixture.query,
            benchmark=fixture.benchmark,
            support_envelope=fixture.support,
            training_run=fixture.training_run,
            context=fixture.context,
        )


def test_stage_coverage_is_nonvacuous_and_never_expands_heldout_partitions(
    training_fixture: TrainingFixture,
) -> None:
    required = tuple(receipt.artifact for receipt in training_fixture.context.artifact_receipts)
    assert any(
        item.artifact_kind is AdmissionArtifactKind.DATASET_SOURCE_ARTIFACT for item in required
    )
    assert any(item.reference_id == "item12-p1-training-evidence" for item in required)
    assert any(item.reference_id == "item12-candidate-factory-interface" for item in required)
    assert not any(
        token in item.reference_id
        for item in required
        for token in ("p2", "p3", "p4", "calibration", "model-selection", "test-outcome")
    )
    training_ids = {
        item.artifact_id for item in training_fixture.training_run.training_evidence_artifacts
    }
    assert "item12-training-plan" in training_ids
    assert "item12-p1-training-evidence" in training_ids
    assert "item12-training-source-selection" not in training_ids
    assert "item12-candidate-fit-receipt" not in training_ids


def test_typed_p1_evidence_and_fit_receipts_reject_heldout_claims_or_wrong_bytes(
    training_fixture: TrainingFixture,
) -> None:
    evidence_payload = training_fixture.context.p1_evidence.model_dump(mode="python")
    evidence_payload["p2_membership_read"] = True
    with pytest.raises(ValidationError):
        P1TrainingEvidence.model_validate(evidence_payload)

    evidence_payload = training_fixture.context.p1_evidence.model_dump(mode="python")
    evidence_payload["accessed_partition_roles"] = (
        BenchmarkPartitionRole.TRAIN,
        BenchmarkPartitionRole.CALIBRATION,
    )
    with pytest.raises(ValidationError, match="only the training partition"):
        P1TrainingEvidence.model_validate(evidence_payload)

    with pytest.raises(ValueError, match="closed model bytes"):
        issue_candidate_fit_receipt(
            receipt_id="wrong-model-fit",
            plan=training_fixture.context.plan,
            source_selection=training_fixture.context.source_selection,
            p1_evidence=training_fixture.context.p1_evidence,
            contained_execution_observation=(
                training_fixture.context.contained_execution_observation
            ),
            training_result=training_fixture.context.fit_receipt.training_result,
            model_artifact=training_fixture.context.fit_receipt.model_artifact,
            observed_model_content=b"substituted-model",
            behavior_manifest_sha256="1" * 64,
            trusted_verifier=training_fixture.context.trusted_verifiers[0],
            issued_at=NOW,
        )


def test_fit_receipt_rejects_source_substitution_in_typed_contained_observation(
    training_fixture: TrainingFixture,
) -> None:
    original = training_fixture.context.contained_execution_observation
    substituted_sha256 = "f" * 64
    substituted_worker = original.worker_observation.model_copy(
        update={
            "expected_source_sha256": substituted_sha256,
            "source_pre_sha256": substituted_sha256,
            "source_post_sha256": substituted_sha256,
        }
    )
    substituted = original.model_copy(update={"worker_observation": substituted_worker})
    with pytest.raises(ValueError, match="used another source"):
        issue_candidate_fit_receipt(
            receipt_id="source-substitution-fit",
            plan=training_fixture.context.plan,
            source_selection=training_fixture.context.source_selection,
            p1_evidence=training_fixture.context.p1_evidence,
            contained_execution_observation=substituted,
            training_result=training_fixture.context.fit_receipt.training_result,
            model_artifact=training_fixture.context.fit_receipt.model_artifact,
            observed_model_content=MODEL_BYTES,
            behavior_manifest_sha256="1" * 64,
            trusted_verifier=training_fixture.context.trusted_verifiers[0],
            issued_at=NOW,
        )


@pytest.mark.parametrize(
    ("field_name", "substituted_value"),
    (
        ("policy_fingerprint", "f" * 64),
        ("runtime_image_digest", "sha256:" + "f" * 64),
    ),
)
def test_contained_observation_rejects_execution_only_identity_substitution(
    training_fixture: TrainingFixture,
    field_name: str,
    substituted_value: str,
) -> None:
    payload = training_fixture.context.contained_execution_observation.model_dump(mode="python")
    execution = dict(payload["execution_observation"])
    execution[field_name] = substituted_value
    payload["execution_observation"] = execution
    with pytest.raises(ValidationError, match="one execution"):
        ContainedTrainingObservation.model_validate(payload)


def test_fit_receipt_rejects_missing_model_entry_in_typed_stage_inventory(
    training_fixture: TrainingFixture,
) -> None:
    original = training_fixture.context.contained_execution_observation
    inventory = StagedTrainingInventory(
        entries=tuple(
            entry
            for entry in original.staged_inventory.entries
            if entry.artifact_role != "model_artifact"
        )
    )
    worker = original.worker_observation.model_copy(
        update={
            "staged_inventory": inventory,
            "staged_tree_sha256": inventory.fingerprint,
            "staged_file_count": len(inventory.entries),
        }
    )
    missing_model = original.model_copy(
        update={
            "staged_inventory": inventory,
            "staged_tree_sha256": inventory.fingerprint,
            "worker_observation": worker,
        }
    )
    with pytest.raises(ValueError, match="exact model_artifact bytes"):
        issue_candidate_fit_receipt(
            receipt_id="missing-model-stage-fit",
            plan=training_fixture.context.plan,
            source_selection=training_fixture.context.source_selection,
            p1_evidence=training_fixture.context.p1_evidence,
            contained_execution_observation=missing_model,
            training_result=training_fixture.context.fit_receipt.training_result,
            model_artifact=training_fixture.context.fit_receipt.model_artifact,
            observed_model_content=MODEL_BYTES,
            behavior_manifest_sha256="1" * 64,
            trusted_verifier=training_fixture.context.trusted_verifiers[0],
            issued_at=NOW,
        )


def test_hmac_or_stage_receipt_substitution_cannot_advance_lifecycle(
    training_fixture: TrainingFixture,
) -> None:
    fixture = training_fixture
    trusted = fixture.context.trusted_verifiers[0]
    wrong_trust = TrustedAdmissionVerifier(
        identity=trusted.identity,
        key_id=trusted.key_id,
        secret=b"x" * 32,
    )
    wrong_hmac_context = replace(fixture.context, trusted_verifiers=(wrong_trust,))
    with pytest.raises(ValueError, match="authentication failed"):
        require_exact_trained_candidate(
            fixture.bundle,
            query=fixture.query,
            benchmark=fixture.benchmark,
            support_envelope=fixture.support,
            training_run=fixture.training_run,
            context=wrong_hmac_context,
        )

    incomplete_context = replace(
        fixture.context,
        artifact_receipts=fixture.context.artifact_receipts[:-1],
    )
    readiness = assess_biological_model_bundle(
        fixture.bundle,
        query=fixture.query,
        benchmark=fixture.benchmark,
        manifests={
            binding.binding_id: fixture.manifest
            for binding in fixture.benchmark.definition.evidence_bindings
        },
        support_envelope=fixture.support,
        training_run=fixture.training_run,
        training_context=incomplete_context,
    )
    assert readiness.training_binding_verified
    assert not readiness.training_artifacts_verified
    assert not readiness.training_interfaces_verified
    assert not readiness.training_result_semantics_verified
    assert not readiness.trained_candidate_verified
    assert readiness.lifecycle_stage is ComponentLifecycleStage.SCAFFOLD
    assert any("stage artifact receipts" in blocker for blocker in readiness.blockers)


def test_plan_rejects_same_code_for_trainer_and_candidate_factory(
    training_fixture: TrainingFixture,
) -> None:
    payload = training_fixture.context.plan.model_dump(mode="python")
    payload["trainer_implementation"]["code_artifact"] = payload[
        "candidate_factory_implementation"
    ]["code_artifact"]
    with pytest.raises(ValidationError, match="distinct"):
        CandidateTrainingPlan.model_validate(payload)

    payload = training_fixture.context.plan.model_dump(mode="python")
    payload["training_partition_roles"] = (BenchmarkPartitionRole.CALIBRATION,)
    with pytest.raises(ValidationError, match="typed training partition role"):
        CandidateTrainingPlan.model_validate(payload)


def test_runtime_training_context_detaches_caller_owned_collections(
    training_fixture: TrainingFixture,
) -> None:
    original = training_fixture.context
    artifact_receipts = list(original.artifact_receipts)
    trusted_verifiers = list(original.trusted_verifiers)
    runtime_interfaces = dict(original.runtime_interfaces)
    detached = TrainingVerificationContext(
        plan=original.plan,
        plan_artifact=original.plan_artifact,
        p1_evidence=original.p1_evidence,
        p1_evidence_artifact=original.p1_evidence_artifact,
        contained_execution_policy=original.contained_execution_policy,
        runtime_image_lock=original.runtime_image_lock,
        training_code_closure=original.training_code_closure,
        training_execution_input_closure=original.training_execution_input_closure,
        contained_execution_observation=original.contained_execution_observation,
        source_selection=original.source_selection,
        fit_receipt=original.fit_receipt,
        query_prerequisite_report=original.query_prerequisite_report,
        artifact_receipts=artifact_receipts,  # type: ignore[arg-type]
        candidate_factory_interface_receipt=(original.candidate_factory_interface_receipt),
        trusted_verifiers=trusted_verifiers,  # type: ignore[arg-type]
        runtime_interfaces=runtime_interfaces,
    )
    artifact_receipts.clear()
    trusted_verifiers.clear()
    runtime_interfaces.clear()
    assert detached.artifact_receipts == original.artifact_receipts
    assert detached.trusted_verifiers == original.trusted_verifiers
    assert detached.runtime_interfaces == original.runtime_interfaces


def test_training_byte_observer_streams_exact_bytes_and_rejects_vacuous_inputs() -> None:
    expected = (sha256(MODEL_BYTES).hexdigest(), len(MODEL_BYTES))
    assert training_contracts._observe_bytes(io.BytesIO(MODEL_BYTES)) == expected
    assert (
        training_contracts._observe_bytes((MODEL_BYTES[:5], MODEL_BYTES[5:17], MODEL_BYTES[17:]))
        == expected
    )

    with pytest.raises(ValueError, match="must not be empty"):
        training_contracts._observe_bytes(io.BytesIO())
    with pytest.raises(ValueError, match="must not be empty"):
        training_contracts._observe_bytes(())
    with pytest.raises(TypeError, match="exact bytes"):
        training_contracts._observe_bytes((bytearray(b"not-exact-bytes"),))  # type: ignore[arg-type]


def test_training_contract_primitive_helpers_reject_noncanonical_lookalikes() -> None:
    with pytest.raises(ValueError, match="canonical nonblank string"):
        training_contracts._canonical_text(" padded", name="test value")
    with pytest.raises(ValueError, match="exact strings"):
        training_contracts._canonical_sha256(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="immutable tuple"):
        training_contracts._canonical_values(["p1-train"], name="partitions")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        training_contracts._canonical_values((), name="partitions", allow_empty=False)
    with pytest.raises(ValueError, match="must be unique"):
        training_contracts._canonical_values(("p1", "p1"), name="partitions")
    with pytest.raises(ValueError, match="must be sorted"):
        training_contracts._canonical_values(("p2", "p1"), name="partitions")
    with pytest.raises(ValueError, match="exact datetime"):
        training_contracts._require_utc("2026-08-10", name="issued at")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        training_contracts._require_utc(datetime(2026, 8, 10), name="issued at")
    with pytest.raises(ValueError, match="must use UTC"):
        training_contracts._require_utc(
            datetime(2026, 8, 10, tzinfo=timezone(timedelta(hours=1))),
            name="issued at",
        )
