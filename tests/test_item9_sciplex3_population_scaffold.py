"""P0 guards for the first narrow, deliberately non-runnable population component."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Never, get_type_hints

import pytest

from cellstate.backends.contracts import (
    BiologicalModelBundleContract,
    BiologicalStagePort,
    BiologicalSupportEnvelope,
    BundleContractKind,
    ComponentLifecycleStage,
    PortDisposition,
)
from cellstate.backends.sciplex3_k562 import (
    SCIPLEX3_K562_BENCHMARK_SHA256,
    SCIPLEX3_K562_BUNDLE_CONTRACT_SHA256,
    SCIPLEX3_K562_MANIFEST_SHA256,
    SCIPLEX3_K562_QUERY_SHA256,
    SCIPLEX3_K562_SUPPORT_ENVELOPE_SHA256,
    PopulationAssayResponseBlockerCode,
    PopulationAssayResponseTask,
    PopulationComponentAccessPurpose,
    PopulationResponseRepresentation,
    SciPlex3K562PopulationAssayResponseScaffold,
)
from cellstate.data.benchmarks import (
    BenchmarkArtifact,
    BenchmarkEvaluationCase,
)
from cellstate.domain.common import CausalStatus, OntologyTerm, canonical_json_bytes
from cellstate.domain.query import StateQuery
from cellstate.errors import ContractViolationError
from cellstate.ports import CellStateEstimator, PopulationAssayResponseModel

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data_manifests/reviewed/sciplex3-k562-24h.json"
QUERY_PATH = ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/state-query.json"
BENCHMARK_PATH = ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1/benchmark-artifact.json"
COMPONENT_DIR = ROOT / "backends/vertical-a/sciplex3-k562-24h-v1"
SUPPORT_PATH = COMPONENT_DIR / "support-envelope.json"
BUNDLE_PATH = COMPONENT_DIR / "bundle-contract.json"


@pytest.fixture(scope="module")
def query() -> StateQuery:
    return StateQuery.model_validate_json(QUERY_PATH.read_bytes())


@pytest.fixture(scope="module")
def benchmark() -> BenchmarkArtifact:
    return BenchmarkArtifact.model_validate_json(BENCHMARK_PATH.read_bytes())


@pytest.fixture(scope="module")
def scaffold() -> SciPlex3K562PopulationAssayResponseScaffold:
    return SciPlex3K562PopulationAssayResponseScaffold.from_repository(ROOT)


def _case(benchmark: BenchmarkArtifact, partition_id: str) -> BenchmarkEvaluationCase:
    case_set = benchmark.definition.evaluation_case_set
    assert case_set is not None
    return next(case for case in case_set.cases if case.partition_id == partition_id)


def _task(
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    *,
    partition_id: str = "p1-train",
    access_purpose: PopulationComponentAccessPurpose = (
        PopulationComponentAccessPurpose.TRAIN_PARAMETERS
    ),
) -> PopulationAssayResponseTask:
    return PopulationAssayResponseTask(
        query=query,
        evaluation_case=_case(benchmark, partition_id),
        access_purpose=access_purpose,
        output_representation=PopulationResponseRepresentation.PREDICTIVE_SAMPLES,
        forecast_causal_status=CausalStatus.PREDICTIVE_ASSOCIATION,
    )


def _codes(report: object, field: str) -> set[PopulationAssayResponseBlockerCode]:
    return {blocker.code for blocker in getattr(report, field)}


def _mutate_case(
    case: BenchmarkEvaluationCase,
    **updates: object,
) -> BenchmarkEvaluationCase:
    payload = case.model_dump(mode="python")
    payload.update(updates)
    return BenchmarkEvaluationCase.model_validate(payload)


def test_checked_in_contracts_are_exact_scaffolds_with_no_runtime_surface() -> None:
    expected_hashes = {
        MANIFEST_PATH: SCIPLEX3_K562_MANIFEST_SHA256,
        QUERY_PATH: SCIPLEX3_K562_QUERY_SHA256,
        BENCHMARK_PATH: SCIPLEX3_K562_BENCHMARK_SHA256,
        SUPPORT_PATH: SCIPLEX3_K562_SUPPORT_ENVELOPE_SHA256,
        BUNDLE_PATH: SCIPLEX3_K562_BUNDLE_CONTRACT_SHA256,
    }
    for path, expected_hash in expected_hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash

    support = BiologicalSupportEnvelope.model_validate_json(SUPPORT_PATH.read_bytes())
    bundle = BiologicalModelBundleContract.model_validate_json(BUNDLE_PATH.read_bytes())
    assert SUPPORT_PATH.read_bytes() == canonical_json_bytes(support.model_dump(mode="json"))
    assert BUNDLE_PATH.read_bytes() == canonical_json_bytes(bundle.model_dump(mode="json"))
    assert support.bundle_kind is BundleContractKind.COMPONENT_SCAFFOLD
    assert support.direct_population_response is not None
    assert support.direct_population_response.inference_inputs == (
        "static_context_and_assigned_action_only"
    )
    assert support.direct_population_response.endpoint_observations == "future_target_only"
    assert not support.runtime_operations
    assert bundle.bundle_kind is BundleContractKind.COMPONENT_SCAFFOLD
    assert bundle.model_artifact is None
    assert bundle.training_run is None
    assert not bundle.validation_evidence
    assert bundle.posterior_schema_id is None
    assert not bundle.operation_implementations

    dispositions = {binding.port: binding.disposition for binding in bundle.ports}
    assert {
        port
        for port, disposition in dispositions.items()
        if disposition is PortDisposition.REQUIRED
    } == set(support.required_ports)
    assert {
        BiologicalStagePort.REFERENCE_PRIOR,
        BiologicalStagePort.INTERVENTION_REALIZATION_MODEL,
        BiologicalStagePort.SUFFICIENCY_EVALUATOR,
        BiologicalStagePort.IDENTIFIABILITY_ANALYZER,
    } <= {
        port
        for port, disposition in dispositions.items()
        if disposition is PortDisposition.UNSUPPORTED
    }
    assert dispositions[BiologicalStagePort.TRANSITION_MODEL] is PortDisposition.NOT_APPLICABLE
    assert (
        dispositions[BiologicalStagePort.EXTRACELLULAR_TRANSPORT_MODEL]
        is PortDisposition.NOT_APPLICABLE
    )
    assert dispositions[BiologicalStagePort.OBSERVATION_MODELS] is PortDisposition.NOT_APPLICABLE
    assert dispositions[BiologicalStagePort.VALUE_OF_INFORMATION_ENGINE] is (
        PortDisposition.NOT_APPLICABLE
    )


def test_exact_scope_preflight_is_still_scaffold_and_never_a_belief_backend(
    scaffold: SciPlex3K562PopulationAssayResponseScaffold,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
) -> None:
    task = _task(query, benchmark)
    report = scaffold.preflight(task)
    assert isinstance(scaffold, PopulationAssayResponseModel)
    assert not isinstance(scaffold, CellStateEstimator)
    assert report.scope_blockers == ()
    assert report.bundle_readiness.lifecycle_stage is ComponentLifecycleStage.SCAFFOLD
    assert report.bundle_readiness.query_binding_verified
    assert report.bundle_readiness.benchmark_binding_verified
    assert report.bundle_readiness.support_envelope_binding_verified
    assert not report.bundle_readiness.training_binding_verified
    assert not report.bundle_readiness.calibration_binding_verified
    assert not report.bundle_readiness.model_selection_binding_verified
    assert not report.bundle_readiness.component_evaluation_complete
    assert not report.bundle_readiness.runtime_registration_allowed
    assert not report.execution_allowed
    assert not report.can_emit_population_response_distribution
    assert not report.can_emit_cell_state_belief
    assert {
        PopulationAssayResponseBlockerCode.BASELINES_NOT_EXECUTED,
        PopulationAssayResponseBlockerCode.BUNDLE_NOT_RUNNABLE,
        PopulationAssayResponseBlockerCode.METRIC_IMPLEMENTATIONS_ABSENT,
        PopulationAssayResponseBlockerCode.MODEL_ARTIFACT_ABSENT,
        PopulationAssayResponseBlockerCode.RUNTIME_SUPPORT_NOT_ADMITTED,
        PopulationAssayResponseBlockerCode.SOURCE_DUPLICATE_AUDIT_UNASSESSED,
        PopulationAssayResponseBlockerCode.VALIDATION_EVIDENCE_ABSENT,
    } <= _codes(report, "readiness_blockers")
    assert not hasattr(scaffold, "estimate")
    assert not hasattr(scaffold, "predict_from_context")


def test_sample_response_fails_closed_before_any_biological_value(
    scaffold: SciPlex3K562PopulationAssayResponseScaffold,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
) -> None:
    with pytest.raises(RuntimeError, match="non-runnable scaffold"):
        scaffold.sample_response(_task(query, benchmark), sample_count=8, seed=9)


@pytest.mark.parametrize(
    "relative_path",
    (
        "data_manifests/reviewed/sciplex3-k562-24h.json",
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/state-query.json",
        "benchmarks/vertical-a/sciplex3-k562-24h-v1/benchmark-artifact.json",
        "backends/vertical-a/sciplex3-k562-24h-v1/support-envelope.json",
        "backends/vertical-a/sciplex3-k562-24h-v1/bundle-contract.json",
    ),
)
def test_every_loaded_contract_hash_fails_closed_on_byte_drift(
    tmp_path: Path,
    relative_path: str,
) -> None:
    required = (
        MANIFEST_PATH,
        QUERY_PATH,
        BENCHMARK_PATH,
        SUPPORT_PATH,
        BUNDLE_PATH,
    )
    target = ROOT / relative_path
    for source in required:
        destination = tmp_path / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source == target:
            destination.write_bytes(source.read_bytes() + b"\n")
        else:
            destination.symlink_to(source)
    with pytest.raises(ContractViolationError, match="SHA-256 drift"):
        SciPlex3K562PopulationAssayResponseScaffold.from_repository(tmp_path)


def test_wrong_query_is_outside_scope(
    scaffold: SciPlex3K562PopulationAssayResponseScaffold,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
) -> None:
    wrong_query = query.model_copy(update={"temporal_resolution_seconds": 2.0})
    task = _task(query, benchmark).model_copy(update={"query": wrong_query})
    assert PopulationAssayResponseBlockerCode.QUERY_MISMATCH in _codes(
        scaffold.preflight(task), "scope_blockers"
    )


@pytest.mark.parametrize(
    ("updates", "expected_code"),
    (
        (
            {"intervention_spec_ids": ("unsupported-compound-99999-nm",)},
            PopulationAssayResponseBlockerCode.ACTION_OR_DOSE_MISMATCH,
        ),
        (
            {"target_output_keys": ("cellstate:viability",)},
            PopulationAssayResponseBlockerCode.TARGET_MISMATCH,
        ),
        (
            {"horizon_name": "72h-endpoint"},
            PopulationAssayResponseBlockerCode.HORIZON_MISMATCH,
        ),
        (
            {"context_id": "static-context--unsupported-plate"},
            PopulationAssayResponseBlockerCode.EVALUATION_CASE_MISMATCH,
        ),
        (
            {"case_id": "well-case--unknown"},
            PopulationAssayResponseBlockerCode.EVALUATION_CASE_UNKNOWN,
        ),
    ),
)
def test_case_action_target_horizon_and_context_are_exact(
    scaffold: SciPlex3K562PopulationAssayResponseScaffold,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    updates: dict[str, object],
    expected_code: PopulationAssayResponseBlockerCode,
) -> None:
    base = _task(query, benchmark)
    task = base.model_copy(
        update={"evaluation_case": _mutate_case(base.evaluation_case, **updates)}
    )
    assert expected_code in _codes(scaffold.preflight(task), "scope_blockers")


@pytest.mark.parametrize(
    ("partition_id", "access_purpose"),
    (
        ("p1-train", PopulationComponentAccessPurpose.TRAIN_PARAMETERS),
        ("p2-calibration", PopulationComponentAccessPurpose.FIT_CALIBRATION),
        (
            "p3-model-selection-validation",
            PopulationComponentAccessPurpose.MODEL_SELECTION,
        ),
        (
            "p4-untouched-test",
            PopulationComponentAccessPurpose.UNTOUCHED_EVALUATION,
        ),
    ),
)
def test_partition_lifecycle_access_is_exact(
    scaffold: SciPlex3K562PopulationAssayResponseScaffold,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    partition_id: str,
    access_purpose: PopulationComponentAccessPurpose,
) -> None:
    report = scaffold.preflight(
        _task(query, benchmark, partition_id=partition_id, access_purpose=access_purpose)
    )
    assert PopulationAssayResponseBlockerCode.PARTITION_ROLE_MISMATCH not in _codes(
        report, "scope_blockers"
    )


@pytest.mark.parametrize(
    "partition_id",
    ("p2-calibration", "p3-model-selection-validation", "p4-untouched-test"),
)
def test_nontraining_partitions_cannot_be_opened_for_parameter_training(
    scaffold: SciPlex3K562PopulationAssayResponseScaffold,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    partition_id: str,
) -> None:
    report = scaffold.preflight(
        _task(
            query,
            benchmark,
            partition_id=partition_id,
            access_purpose=PopulationComponentAccessPurpose.TRAIN_PARAMETERS,
        )
    )
    codes = _codes(report, "scope_blockers")
    assert PopulationAssayResponseBlockerCode.PARTITION_ROLE_MISMATCH in codes
    assert PopulationAssayResponseBlockerCode.TRAINING_PARTITION_VIOLATION in codes
    if partition_id == "p4-untouched-test":
        assert PopulationAssayResponseBlockerCode.UNTOUCHED_TEST_MUTATION in codes


@pytest.mark.parametrize(
    ("updates", "expected_code"),
    (
        (
            {
                "pre_cutoff_evidence_modalities": (
                    OntologyTerm(
                        label="single nucleus RNA sequencing",
                        identifier="EFO:0009809",
                        namespace="EFO",
                    ),
                )
            },
            PopulationAssayResponseBlockerCode.ENDPOINT_TARGET_LEAKAGE,
        ),
        (
            {"output_representation": PopulationResponseRepresentation.POINT_ESTIMATE},
            PopulationAssayResponseBlockerCode.POINT_ONLY_OUTPUT_UNSUPPORTED,
        ),
        (
            {"forecast_causal_status": CausalStatus.IDENTIFIED_POPULATION_EFFECT},
            PopulationAssayResponseBlockerCode.CAUSAL_STATUS_OVERCLAIM,
        ),
        (
            {"requested_environment_keys": ("oxygen",)},
            PopulationAssayResponseBlockerCode.ENVIRONMENT_UNSUPPORTED,
        ),
        (
            {"transport_requested": True},
            PopulationAssayResponseBlockerCode.TRANSPORT_UNSUPPORTED,
        ),
        (
            {"identified_intervention_realization_required": True},
            PopulationAssayResponseBlockerCode.INTERVENTION_REALIZATION_UNIDENTIFIED,
        ),
        (
            {"survival_or_viability_interpretation_required": True},
            PopulationAssayResponseBlockerCode.SURVIVAL_VIABILITY_UNSUPPORTED,
        ),
        (
            {"current_hidden_state_inference_required": True},
            PopulationAssayResponseBlockerCode.CURRENT_HIDDEN_STATE_UNSUPPORTED,
        ),
    ),
)
def test_endpoint_leakage_and_scientific_overclaims_fail_closed(
    scaffold: SciPlex3K562PopulationAssayResponseScaffold,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    updates: dict[str, object],
    expected_code: PopulationAssayResponseBlockerCode,
) -> None:
    task = _task(query, benchmark).model_copy(update=updates)
    assert expected_code in _codes(scaffold.preflight(task), "scope_blockers")


def test_scaffold_has_no_constructible_public_response_contract() -> None:
    import cellstate.backends as backends

    assert not hasattr(backends, "PopulationAssayResponseSamples")
    assert not hasattr(backends, "PopulationAssayResponsePredictionProvenance")
    assert "PopulationAssayResponseSamples" not in backends.__all__
    assert "PopulationAssayResponsePredictionProvenance" not in backends.__all__
    assert not (
        ROOT / "schemas/experimental/population-assay-response-samples.schema.json"
    ).exists()
    return_type = get_type_hints(SciPlex3K562PopulationAssayResponseScaffold.sample_response)[
        "return"
    ]
    assert return_type is Never
    assert (
        inspect.signature(
            SciPlex3K562PopulationAssayResponseScaffold.sample_response
        ).return_annotation
        == "Never"
    )
