from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

import cellstate
from cellstate.backends import (
    AdmissionReceiptBatchReport,
    ArtifactResolutionReceipt,
    BiologicalModelBundleContract,
    BiologicalSupportEnvelope,
    BundleReadiness,
    CandidateFitReceipt,
    CandidateTrainingPlan,
    ExecutionSourceSelectionReceipt,
    LoadedInterfaceReceipt,
    P1TrainingEvidence,
    PopulationAssayResponsePreflight,
    PopulationAssayResponseTask,
    QueryDerivedPrerequisiteReport,
    TrainedCandidateVerification,
    TrainingRunBinding,
    TrainingSourceSelectionReceipt,
    ValidationEvidenceBinding,
    ValidationResultManifest,
    ValidationResultReceiptBatch,
    ValidationResultVerificationReceipt,
)
from cellstate.data import BenchmarkArtifact, DatasetManifest, RepresentabilityProof
from cellstate.domain import (
    BeliefDiagnostics,
    BeliefSubject,
    CausalStatus,
    CausalSupportReport,
    CellHistory,
    CellStateBelief,
    CompiledStateSpecification,
    EstimateCellStateRequest,
    EvolutionScenario,
    InterventionObjective,
    InterventionPlan,
    MeasurementDecisionRequest,
    MeasurementRecommendation,
    QueryReadinessReport,
    StateDistribution,
    StateForecast,
    StateQuery,
    SubjectSpecification,
    SupportReport,
    TargetPrediction,
)

SCHEMA_ROOT = Path(__file__).resolve().parents[1] / "schemas"
V2_MODELS: dict[str, type[BaseModel]] = {
    "belief-subject.schema.json": BeliefSubject,
    "subject-specification.schema.json": SubjectSpecification,
    "state-query.schema.json": StateQuery,
    "compiled-state-specification.schema.json": CompiledStateSpecification,
    "cell-history.schema.json": CellHistory,
    "estimate-cell-state-request.schema.json": EstimateCellStateRequest,
    "cell-state-belief.schema.json": CellStateBelief,
    "evolution-scenario.schema.json": EvolutionScenario,
    "state-forecast.schema.json": StateForecast,
    "intervention-objective.schema.json": InterventionObjective,
    "intervention-plan.schema.json": InterventionPlan,
    "measurement-decision-request.schema.json": MeasurementDecisionRequest,
    "measurement-recommendation.schema.json": MeasurementRecommendation,
}


def test_public_v2_models_generate_json_schema() -> None:
    for model in (
        BeliefSubject,
        SubjectSpecification,
        StateQuery,
        CompiledStateSpecification,
        CellHistory,
        EstimateCellStateRequest,
        CellStateBelief,
        EvolutionScenario,
        StateForecast,
        InterventionObjective,
        InterventionPlan,
        MeasurementDecisionRequest,
        MeasurementRecommendation,
        BeliefDiagnostics,
        SupportReport,
        QueryReadinessReport,
        CausalSupportReport,
        TargetPrediction,
    ):
        schema = model.model_json_schema()
        assert schema["type"] == "object"
        assert "$defs" in schema

    distribution_schema = TypeAdapter(StateDistribution).json_schema()
    assert "oneOf" in distribution_schema
    assert "$defs" in distribution_schema


def test_v2_contracts_are_exported_from_domain_and_package_root() -> None:
    expected = {
        "BeliefSubject": BeliefSubject,
        "SubjectSpecification": SubjectSpecification,
        "StateQuery": StateQuery,
        "CompiledStateSpecification": CompiledStateSpecification,
        "StateDistribution": StateDistribution,
        "BeliefDiagnostics": BeliefDiagnostics,
        "SupportReport": SupportReport,
        "QueryReadinessReport": QueryReadinessReport,
        "CausalStatus": CausalStatus,
        "CausalSupportReport": CausalSupportReport,
        "StateForecast": StateForecast,
        "TargetPrediction": TargetPrediction,
        "InterventionPlan": InterventionPlan,
        "MeasurementDecisionRequest": MeasurementDecisionRequest,
        "MeasurementRecommendation": MeasurementRecommendation,
    }
    for name, contract in expected.items():
        assert getattr(cellstate, name) is contract
        assert name in cellstate.__all__

    assert {
        "estimate_cell_state",
        "evolve_cell_state",
        "choose_intervention",
        "recommend_next_measurement",
    } <= set(cellstate.__all__)


def test_checked_in_v1_schemas_match_the_immutable_hash_inventory() -> None:
    v1_root = SCHEMA_ROOT / "v1"
    hash_lines = (v1_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    expected_hashes = {
        filename: digest for line in hash_lines for digest, filename in (line.split(maxsplit=1),)
    }

    checked_in_schemas = {path.name for path in v1_root.glob("*.schema.json")}
    assert checked_in_schemas == set(expected_hashes)
    for filename, expected_digest in expected_hashes.items():
        actual_digest = hashlib.sha256((v1_root / filename).read_bytes()).hexdigest()
        assert actual_digest == expected_digest


def test_checked_in_v2_schemas_are_current() -> None:
    v2_root = SCHEMA_ROOT / "v2"
    checked_in_schemas = {path.name for path in v2_root.glob("*.schema.json")}
    assert checked_in_schemas == set(V2_MODELS)
    for filename, model in V2_MODELS.items():
        checked_in = json.loads((v2_root / filename).read_text(encoding="utf-8"))
        assert checked_in == model.model_json_schema()


def test_checked_in_experimental_schemas_are_current_and_warn_consumers() -> None:
    models: dict[str, type[BaseModel]] = {
        "admission-receipt-batch.schema.json": AdmissionReceiptBatchReport,
        "artifact-resolution-receipt.schema.json": ArtifactResolutionReceipt,
        "benchmark-artifact.schema.json": BenchmarkArtifact,
        "biological-model-bundle.schema.json": BiologicalModelBundleContract,
        "biological-support-envelope.schema.json": BiologicalSupportEnvelope,
        "bundle-readiness.schema.json": BundleReadiness,
        "candidate-fit-receipt.schema.json": CandidateFitReceipt,
        "candidate-training-plan.schema.json": CandidateTrainingPlan,
        "dataset-manifest.schema.json": DatasetManifest,
        "execution-source-selection-receipt.schema.json": ExecutionSourceSelectionReceipt,
        "loaded-interface-receipt.schema.json": LoadedInterfaceReceipt,
        "population-assay-response-preflight.schema.json": PopulationAssayResponsePreflight,
        "population-assay-response-task.schema.json": PopulationAssayResponseTask,
        "p1-training-evidence.schema.json": P1TrainingEvidence,
        "query-derived-prerequisites.schema.json": QueryDerivedPrerequisiteReport,
        "representability-proof.schema.json": RepresentabilityProof,
        "training-run-binding.schema.json": TrainingRunBinding,
        "training-source-selection-receipt.schema.json": TrainingSourceSelectionReceipt,
        "trained-candidate-verification.schema.json": TrainedCandidateVerification,
        "validation-evidence-binding.schema.json": ValidationEvidenceBinding,
        "validation-result-manifest.schema.json": ValidationResultManifest,
        "validation-result-receipt-batch.schema.json": ValidationResultReceiptBatch,
        "validation-result-receipt.schema.json": ValidationResultVerificationReceipt,
    }
    experimental_root = SCHEMA_ROOT / "experimental"
    assert {path.name for path in experimental_root.glob("*.schema.json")} == set(models)
    for filename, model in models.items():
        checked_in = json.loads((experimental_root / filename).read_text(encoding="utf-8"))
        assert checked_in == model.model_json_schema()

    dataset_schema = json.loads(
        (experimental_root / "dataset-manifest.schema.json").read_text(encoding="utf-8")
    )
    assert "Python model" in dataset_schema["$comment"]
