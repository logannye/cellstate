#!/usr/bin/env python3
"""Build the canonical non-runnable sci-Plex3 K562 component bundle artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from cellstate.backends.contracts import (
    BiologicalModelBundleContract,
    BiologicalStagePort,
    BiologicalSupportEnvelope,
    BundleContractKind,
    BundleContractReference,
    DirectPopulationResponseSemantics,
    ModelPortBinding,
    PortDisposition,
    ValidationEvidenceKind,
    ValidationEvidenceRequirement,
)
from cellstate.data.benchmarks import (
    BenchmarkArtifact,
    BenchmarkPartitionRole,
    ContentAddressedArtifact,
)
from cellstate.domain.common import canonical_json_bytes
from cellstate.domain.query import StateQuery

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPONENT_DIRECTORY = REPOSITORY_ROOT / "backends/vertical-a/sciplex3-k562-24h-v1"
SUPPORT_ENVELOPE_PATH = COMPONENT_DIRECTORY / "support-envelope.json"
BUNDLE_PATH = COMPONENT_DIRECTORY / "bundle-contract.json"
QUERY_PATH = REPOSITORY_ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/state-query.json"
BENCHMARK_PATH = (
    REPOSITORY_ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/benchmark-artifact.json"
)
RAW_BASE = "https://raw.githubusercontent.com/logannye/cellstate/main"

REQUIRED_PORTS = frozenset(
    {
        BiologicalStagePort.ACTION_CONTEXT_ENCODER,
        BiologicalStagePort.ARTIFACT_PROVENANCE_WRITER,
        BiologicalStagePort.EXACT_ARTIFACT_RESOLVER,
        BiologicalStagePort.POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL,
        BiologicalStagePort.QUERY_SCOPE_VALIDATOR,
        BiologicalStagePort.STRICT_SUPPORT_OOD_GATE,
        BiologicalStagePort.TRAIN_CAL_DATA_LOADER,
        BiologicalStagePort.UNCERTAINTY_CALIBRATOR,
    }
)
UNSUPPORTED_PORTS = frozenset(
    {
        BiologicalStagePort.IDENTIFIABILITY_ANALYZER,
        BiologicalStagePort.INTERVENTION_REALIZATION_MODEL,
        BiologicalStagePort.REFERENCE_PRIOR,
        BiologicalStagePort.SUFFICIENCY_EVALUATOR,
    }
)

PORT_RATIONALES = {
    BiologicalStagePort.ACTION_CONTEXT_ENCODER: (
        "An exact compound-dose and plate-context encoder is mandatory but not implemented."
    ),
    BiologicalStagePort.ARTIFACT_PROVENANCE_WRITER: (
        "A prediction provenance writer bound to model, training, validation, query, benchmark, "
        "case, seed, and ordered features is mandatory but not implemented."
    ),
    BiologicalStagePort.CELL_INTERACTION_MODEL: (
        "The endpoint-only component declares no neighborhood or interaction domain."
    ),
    BiologicalStagePort.DIVISION_AND_INHERITANCE_MODEL: (
        "No lineage identity, division path, or inheritance target exists in this component."
    ),
    BiologicalStagePort.EVIDENCE_TRANSFER_MODELS: (
        "No pre-cutoff cross-subject molecular evidence is transferred into the endpoint task."
    ),
    BiologicalStagePort.EXACT_ARTIFACT_RESOLVER: (
        "An admitted exact-byte resolver is mandatory but has no bundle implementation binding."
    ),
    BiologicalStagePort.EXTRACELLULAR_TRANSPORT_MODEL: (
        "The frozen query forbids environment and cross-domain transport."
    ),
    BiologicalStagePort.FUNCTIONAL_DECODERS: (
        "Recovered-nucleus RNA is the sole target and cannot be decoded as viability or function."
    ),
    BiologicalStagePort.IDENTIFIABILITY_ANALYZER: (
        "No hidden t=0 state is estimated, so its dimension-level identifiability is unsupported."
    ),
    BiologicalStagePort.INTERVENTION_REALIZATION_MODEL: (
        "Randomized assignment is observed, but intracellular exposure and engagement remain "
        "unknown."
    ),
    BiologicalStagePort.MECHANISTIC_CONSTRAINTS: (
        "The direct endpoint component declares no mechanistic latent-state constraints."
    ),
    BiologicalStagePort.MODEL_ENSEMBLE: (
        "No trained response model exists from which an ensemble could be composed."
    ),
    BiologicalStagePort.OBSERVATION_MODELS: (
        "Future single-nucleus RNA is the target and may not be relabeled as pre-cutoff evidence."
    ),
    BiologicalStagePort.OOD_DETECTOR: (
        "The full estimator OOD interface is outside this component; its exact and learned support "
        "duties belong to strict_support_ood_gate."
    ),
    BiologicalStagePort.POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL: (
        "The required predictive count-distribution model and weights are absent."
    ),
    BiologicalStagePort.POSTERIOR_INFERENCE_ENGINE: (
        "This direct endpoint task does not infer a current hidden-state posterior."
    ),
    BiologicalStagePort.QUERY_COMPILER: (
        "The component consumes one exact frozen endpoint task and compiles no latent belief state."
    ),
    BiologicalStagePort.QUERY_SCOPE_VALIDATOR: (
        "An admitted exact-scope validator is mandatory but has no bundle implementation binding."
    ),
    BiologicalStagePort.REFERENCE_PRIOR: (
        "Static plate context is not a t=0 state prior; a biological prior is unsupported here."
    ),
    BiologicalStagePort.STRICT_SUPPORT_OOD_GATE: (
        "A validated exact-support and learned OOD gate is mandatory but absent."
    ),
    BiologicalStagePort.SUFFICIENCY_EVALUATOR: (
        "One context-to-endpoint experiment cannot establish hidden-state sufficiency."
    ),
    BiologicalStagePort.TRAIN_CAL_DATA_LOADER: (
        "A loader enforcing p1 training, p2 calibration, p3 selection, and locked p4 use is absent."
    ),
    BiologicalStagePort.TRANSITION_MODEL: (
        "The component predicts one 24-hour endpoint directly and does not expose state dynamics."
    ),
    BiologicalStagePort.UNCERTAINTY_CALIBRATOR: (
        "Predictive sample calibration on p2 and locked evaluation on p4 are mandatory but absent."
    ),
    BiologicalStagePort.VALUE_OF_INFORMATION_ENGINE: (
        "No assay-selection or decision-utility operation is in component scope."
    ),
}


def _artifact(path: Path, *, artifact_id: str) -> ContentAddressedArtifact:
    payload = path.read_bytes()
    relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
    return ContentAddressedArtifact(
        artifact_id=artifact_id,
        uri=f"{RAW_BASE}/{relative_path}",
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
        media_type="application/json",
    )


def _contract_reference(
    *,
    contract_id: str,
    contract_version: str,
    path: Path,
    artifact_id: str,
) -> BundleContractReference:
    return BundleContractReference(
        contract_id=contract_id,
        contract_version=contract_version,
        artifact=_artifact(path, artifact_id=artifact_id),
    )


def build() -> tuple[BiologicalSupportEnvelope, BiologicalModelBundleContract]:
    query = StateQuery.model_validate_json(QUERY_PATH.read_bytes())
    benchmark = BenchmarkArtifact.model_validate_json(BENCHMARK_PATH.read_bytes())
    query_binding = benchmark.definition.query
    query_reference = _contract_reference(
        contract_id=query_binding.query_id,
        contract_version=query_binding.query_version,
        path=QUERY_PATH,
        artifact_id="sciplex3-k562-frozen-state-query-v2",
    )
    benchmark_reference = _contract_reference(
        contract_id=benchmark.definition.benchmark_id,
        contract_version=benchmark.definition.benchmark_version,
        path=BENCHMARK_PATH,
        artifact_id="sciplex3-k562-component-benchmark",
    )
    if query_reference.artifact.sha256 != query.fingerprint:
        raise ValueError("query file does not contain exact canonical StateQuery bytes")
    if benchmark_reference.artifact.sha256 != benchmark.fingerprint:
        raise ValueError("benchmark file does not contain exact canonical artifact bytes")

    support_envelope = BiologicalSupportEnvelope(
        envelope_id="vertical-a.sciplex3-k562-24h.population-response-envelope",
        envelope_version="0.1.0-scaffold",
        bundle_kind=BundleContractKind.COMPONENT_SCAFFOLD,
        query=query_reference,
        benchmark=benchmark_reference,
        direct_population_response=DirectPopulationResponseSemantics(),
        runtime_operations=(),
        required_ports=tuple(sorted(REQUIRED_PORTS, key=lambda port: port.value)),
        required_validation_evidence=tuple(
            sorted(
                (
                    ValidationEvidenceRequirement(
                        evidence_id="sciplex3-k562-calibration-evidence",
                        evidence_kind=ValidationEvidenceKind.UNCERTAINTY_CALIBRATION,
                        partition_roles=(BenchmarkPartitionRole.CALIBRATION,),
                    ),
                    ValidationEvidenceRequirement(
                        evidence_id="sciplex3-k562-locked-component-evaluation",
                        evidence_kind=ValidationEvidenceKind.LOCKED_COMPONENT_EVALUATION,
                        partition_roles=(BenchmarkPartitionRole.UNTOUCHED_TEST,),
                    ),
                    ValidationEvidenceRequirement(
                        evidence_id="sciplex3-k562-model-selection-evidence",
                        evidence_kind=ValidationEvidenceKind.MODEL_SELECTION,
                        partition_roles=(BenchmarkPartitionRole.MODEL_SELECTION_VALIDATION,),
                    ),
                    ValidationEvidenceRequirement(
                        evidence_id="sciplex3-k562-support-ood-evidence",
                        evidence_kind=ValidationEvidenceKind.SUPPORT_OOD_VALIDATION,
                        partition_roles=(
                            BenchmarkPartitionRole.CALIBRATION,
                            BenchmarkPartitionRole.MODEL_SELECTION_VALIDATION,
                        ),
                    ),
                ),
                key=lambda requirement: requirement.evidence_id,
            )
        ),
        notes=tuple(
            sorted(
                (
                    "Endpoint output is a predictive distribution of raw integer UMI vectors on "
                    "the exact ordered 2,000-feature panel.",
                    "Forecast causal status is predictive_association and realized intracellular "
                    "exposure remains unknown.",
                    "No public estimate, evolution, planning, or measurement operation is "
                    "registered by this component scaffold.",
                    "Static plate and assigned-action context are inputs, not a t=0 hidden-state "
                    "prior or endpoint-response lookup.",
                )
            )
        ),
    )
    support_bytes = canonical_json_bytes(support_envelope.model_dump(mode="json"))
    support_reference = BundleContractReference(
        contract_id=support_envelope.envelope_id,
        contract_version=support_envelope.envelope_version,
        artifact=ContentAddressedArtifact(
            artifact_id="sciplex3-k562-population-response-support-envelope",
            uri=f"{RAW_BASE}/{SUPPORT_ENVELOPE_PATH.relative_to(REPOSITORY_ROOT).as_posix()}",
            sha256=hashlib.sha256(support_bytes).hexdigest(),
            byte_count=len(support_bytes),
            media_type="application/json",
        ),
    )

    port_bindings = tuple(
        ModelPortBinding(
            port=port,
            disposition=(
                PortDisposition.REQUIRED
                if port in REQUIRED_PORTS
                else (
                    PortDisposition.UNSUPPORTED
                    if port in UNSUPPORTED_PORTS
                    else PortDisposition.NOT_APPLICABLE
                )
            ),
            rationale=(PORT_RATIONALES[port],),
        )
        for port in sorted(BiologicalStagePort, key=lambda item: item.value)
    )
    bundle = BiologicalModelBundleContract(
        bundle_id="vertical-a.sciplex3-k562-24h.population-response",
        bundle_version="0.1.0-scaffold",
        bundle_kind=BundleContractKind.COMPONENT_SCAFFOLD,
        description=(
            "Non-runnable scaffold for an exact K562 well-context and assigned-action to 24-hour "
            "recovered-nucleus population assay-response distribution component."
        ),
        posterior_schema_id=None,
        query=query_reference,
        benchmark=benchmark_reference,
        support_envelope=support_reference,
        model_artifact=None,
        training_run=None,
        validation_evidence=(),
        ports=port_bindings,
        operation_implementations=(),
    )
    return support_envelope, bundle


def _emit(path: Path, payload: bytes, *, check: bool) -> None:
    if check:
        if not path.exists() or path.read_bytes() != payload:
            raise SystemExit(f"generated component artifact is stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    support_envelope, bundle = build()
    _emit(
        SUPPORT_ENVELOPE_PATH,
        canonical_json_bytes(support_envelope.model_dump(mode="json")),
        check=args.check,
    )
    _emit(
        BUNDLE_PATH,
        canonical_json_bytes(bundle.model_dump(mode="json")),
        check=args.check,
    )
    print(f"bundle_fingerprint {bundle.fingerprint}")
    print("lifecycle_stage scaffold")
    print("runtime_operations 0")


if __name__ == "__main__":
    main()
