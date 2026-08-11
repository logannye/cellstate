"""Regenerate checked-in JSON Schemas for public boundary models."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

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
    BeliefSubject,
    CellHistory,
    CellStateBelief,
    CompiledStateSpecification,
    EstimateCellStateRequest,
    EvolutionScenario,
    InterventionObjective,
    InterventionPlan,
    MeasurementDecisionRequest,
    MeasurementRecommendation,
    StateForecast,
    StateQuery,
    SubjectSpecification,
)

OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "schemas"
MODELS: dict[Path, type[BaseModel]] = {
    Path("experimental/admission-receipt-batch.schema.json"): AdmissionReceiptBatchReport,
    Path("experimental/artifact-resolution-receipt.schema.json"): ArtifactResolutionReceipt,
    Path("experimental/benchmark-artifact.schema.json"): BenchmarkArtifact,
    Path("experimental/biological-model-bundle.schema.json"): BiologicalModelBundleContract,
    Path("experimental/biological-support-envelope.schema.json"): BiologicalSupportEnvelope,
    Path("experimental/bundle-readiness.schema.json"): BundleReadiness,
    Path("experimental/candidate-fit-receipt.schema.json"): CandidateFitReceipt,
    Path("experimental/candidate-training-plan.schema.json"): CandidateTrainingPlan,
    Path("experimental/dataset-manifest.schema.json"): DatasetManifest,
    Path("experimental/execution-source-selection-receipt.schema.json"): (
        ExecutionSourceSelectionReceipt
    ),
    Path("experimental/loaded-interface-receipt.schema.json"): LoadedInterfaceReceipt,
    Path("experimental/population-assay-response-preflight.schema.json"): (
        PopulationAssayResponsePreflight
    ),
    Path("experimental/population-assay-response-task.schema.json"): PopulationAssayResponseTask,
    Path("experimental/p1-training-evidence.schema.json"): P1TrainingEvidence,
    Path("experimental/query-derived-prerequisites.schema.json"): (QueryDerivedPrerequisiteReport),
    Path("experimental/representability-proof.schema.json"): RepresentabilityProof,
    Path("experimental/training-run-binding.schema.json"): TrainingRunBinding,
    Path("experimental/training-source-selection-receipt.schema.json"): (
        TrainingSourceSelectionReceipt
    ),
    Path("experimental/trained-candidate-verification.schema.json"): (TrainedCandidateVerification),
    Path("experimental/validation-evidence-binding.schema.json"): ValidationEvidenceBinding,
    Path("experimental/validation-result-manifest.schema.json"): ValidationResultManifest,
    Path("experimental/validation-result-receipt.schema.json"): (
        ValidationResultVerificationReceipt
    ),
    Path("experimental/validation-result-receipt-batch.schema.json"): (
        ValidationResultReceiptBatch
    ),
    Path("v2/belief-subject.schema.json"): BeliefSubject,
    Path("v2/subject-specification.schema.json"): SubjectSpecification,
    Path("v2/state-query.schema.json"): StateQuery,
    Path("v2/compiled-state-specification.schema.json"): CompiledStateSpecification,
    Path("v2/cell-history.schema.json"): CellHistory,
    Path("v2/estimate-cell-state-request.schema.json"): EstimateCellStateRequest,
    Path("v2/cell-state-belief.schema.json"): CellStateBelief,
    Path("v2/evolution-scenario.schema.json"): EvolutionScenario,
    Path("v2/state-forecast.schema.json"): StateForecast,
    Path("v2/intervention-objective.schema.json"): InterventionObjective,
    Path("v2/intervention-plan.schema.json"): InterventionPlan,
    Path("v2/measurement-decision-request.schema.json"): MeasurementDecisionRequest,
    Path("v2/measurement-recommendation.schema.json"): MeasurementRecommendation,
}


def main() -> None:
    for relative_path, model in MODELS.items():
        if relative_path.parts[0] not in {"experimental", "v2"}:
            raise RuntimeError("schema export may write only experimental and v2 contracts")
        destination = OUTPUT_ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        destination.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
