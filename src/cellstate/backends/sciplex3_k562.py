"""Fail-closed scaffold for the narrow sci-Plex3 K562 population endpoint component.

This module does not estimate hidden state.  Its eventual computation is limited to predictive
samples from the raw 2,000-feature recovered-nucleus UMI distribution at 24 hours, conditional on
one exact source-well population context and its assigned compound-dose (or matched no-action
control).  Until training, calibration, executable benchmarks, and admission exist, every sampling
attempt raises before reading or returning biological values.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path
from typing import Literal, Never

from pydantic import ConfigDict, Field, field_validator, model_validator

from cellstate.backends.contracts import (
    BiologicalExecutionBlockedError,
    BiologicalModelBundleContract,
    BiologicalSupportEnvelope,
    BundleReadiness,
    assess_biological_model_bundle,
)
from cellstate.data.benchmarks import (
    AuditCheckStatus,
    BaselineRunStatus,
    BenchmarkArtifact,
    BenchmarkEvaluationCase,
    BenchmarkPartitionRole,
    LeakageCheckKind,
    SpecificationOnlyImplementationBinding,
    verify_benchmark_artifact,
)
from cellstate.data.manifests import DatasetManifest
from cellstate.domain.common import (
    CausalStatus,
    OntologyTerm,
    SchemaModel,
    canonical_fingerprint,
    canonical_json_bytes,
)
from cellstate.domain.query import StateQuery, SystemBoundary
from cellstate.errors import ContractViolationError

SCIPLEX3_K562_MANIFEST_SHA256 = "6248e63237a4c0c7ae53538666a1294cf1108569792eb54702ec15f439d9cb31"
SCIPLEX3_K562_QUERY_SHA256 = "d0fa67f31a8ea1d7b2e8839dfe7629fd6f359ea7eed4f6d336e2cd1d8813971e"
SCIPLEX3_K562_BENCHMARK_SHA256 = "97bfb8f00f9efd93ad19635ce1a843a126c3c1b23ae6002102353c5e3bded76e"
SCIPLEX3_K562_TARGET_KEY = "cellstate:sciplex3-k562-24h-train-2000-raw-umi-distribution"
SCIPLEX3_K562_HORIZON_NAME = "24h-endpoint"
SCIPLEX3_K562_FEATURE_COUNT = 2_000
SCIPLEX3_K562_ENDPOINT_MODALITY_KEY = "efo:0009809"
SCIPLEX3_K562_ORDERED_FEATURE_KEYS_SHA256 = (
    "8b9e0e71d9bfc79a5e1db29f73bc30006bf117a29db8abf63b1607403629401f"
)
SCIPLEX3_K562_TARGET_VALUE_SCHEMA_SHA256 = (
    "b2463271246eca932824ad4d0089aaf3c924afcedec865dec8e04c4bbf7b23e2"
)
SCIPLEX3_K562_SUPPORT_ENVELOPE_SHA256 = (
    "17aa440c14b40981f97358119085f44b2ffeb9bed75ba322114ffe2c1c53dd9f"
)
SCIPLEX3_K562_BUNDLE_CONTRACT_SHA256 = (
    "69eddd15eb87b167ea0ef484d54234f6af5ebd6b353ad4aa72a96c2dca3f6343"
)

_MANIFEST_PATH = Path("data_manifests/reviewed/sciplex3-k562-24h.json")
_BENCHMARK_DIRECTORY = Path("benchmarks/vertical-a/sciplex3-k562-24h-v1")
_QUERY_PATH = _BENCHMARK_DIRECTORY / "state-query.json"
_BENCHMARK_PATH = _BENCHMARK_DIRECTORY / "benchmark-artifact.json"
_COMPONENT_DIRECTORY = Path("backends/vertical-a/sciplex3-k562-24h-v1")
_SUPPORT_ENVELOPE_PATH = _COMPONENT_DIRECTORY / "support-envelope.json"
_BUNDLE_PATH = _COMPONENT_DIRECTORY / "bundle-contract.json"


class PopulationResponseRepresentation(StrEnum):
    """Output forms understood by this direct endpoint component."""

    POINT_ESTIMATE = "point_estimate"
    PREDICTIVE_SAMPLES = "predictive_samples"


class PopulationComponentAccessPurpose(StrEnum):
    """Exact lifecycle use of one frozen benchmark partition."""

    FIT_CALIBRATION = "fit_calibration"
    MODEL_SELECTION = "model_selection"
    TRAIN_PARAMETERS = "train_parameters"
    UNTOUCHED_EVALUATION = "untouched_evaluation"


class PopulationAssayResponseBlockerCode(StrEnum):
    """Machine-readable reasons that keep this component fail closed."""

    ACTION_OR_DOSE_MISMATCH = "action_or_dose_mismatch"
    BASELINES_NOT_EXECUTED = "baselines_not_executed"
    BUNDLE_NOT_RUNNABLE = "bundle_not_runnable"
    CAUSAL_STATUS_OVERCLAIM = "causal_status_overclaim"
    CURRENT_HIDDEN_STATE_UNSUPPORTED = "current_hidden_state_unsupported"
    ENDPOINT_TARGET_LEAKAGE = "endpoint_target_leakage"
    ENVIRONMENT_UNSUPPORTED = "environment_unsupported"
    EVALUATION_CASE_MISMATCH = "evaluation_case_mismatch"
    EVALUATION_CASE_UNKNOWN = "evaluation_case_unknown"
    HORIZON_MISMATCH = "horizon_mismatch"
    INTERVENTION_REALIZATION_UNIDENTIFIED = "intervention_realization_unidentified"
    METRIC_IMPLEMENTATIONS_ABSENT = "metric_implementations_absent"
    MODEL_ARTIFACT_ABSENT = "model_artifact_absent"
    PARTITION_ROLE_MISMATCH = "partition_role_mismatch"
    POINT_ONLY_OUTPUT_UNSUPPORTED = "point_only_output_unsupported"
    PRE_CUTOFF_EVIDENCE_UNSUPPORTED = "pre_cutoff_evidence_unsupported"
    QUERY_MISMATCH = "query_mismatch"
    RUNTIME_SUPPORT_NOT_ADMITTED = "runtime_support_not_admitted"
    SOURCE_DUPLICATE_AUDIT_UNASSESSED = "source_duplicate_audit_unassessed"
    SURVIVAL_VIABILITY_UNSUPPORTED = "survival_viability_unsupported"
    TARGET_MISMATCH = "target_mismatch"
    TRAINING_PARTITION_VIOLATION = "training_partition_violation"
    TRANSPORT_UNSUPPORTED = "transport_unsupported"
    UNTOUCHED_TEST_MUTATION = "untouched_test_mutation"
    VALIDATION_EVIDENCE_ABSENT = "validation_evidence_absent"


class _ScaffoldModel(SchemaModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
        strict=True,
    )


class PopulationAssayResponseTask(_ScaffoldModel):
    """One exact direct population-response request at the component boundary."""

    query: StateQuery
    evaluation_case: BenchmarkEvaluationCase
    access_purpose: PopulationComponentAccessPurpose
    pre_cutoff_evidence_modalities: tuple[OntologyTerm, ...] = ()
    output_representation: PopulationResponseRepresentation
    forecast_causal_status: CausalStatus
    requested_environment_keys: tuple[str, ...] = ()
    transport_requested: bool = False
    identified_intervention_realization_required: bool = False
    survival_or_viability_interpretation_required: bool = False
    current_hidden_state_inference_required: bool = False

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))

    @field_validator("pre_cutoff_evidence_modalities")
    @classmethod
    def evidence_modalities_are_canonical(
        cls,
        values: tuple[OntologyTerm, ...],
    ) -> tuple[OntologyTerm, ...]:
        keys = tuple(term.key for term in values)
        if len(keys) != len(set(keys)) or keys != tuple(sorted(keys)):
            raise ValueError("pre-cutoff evidence modalities must be unique and key-sorted")
        return values

    @field_validator("requested_environment_keys")
    @classmethod
    def environment_keys_are_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if (
            any(not value.strip() or value != value.strip() for value in values)
            or len(values) != len(set(values))
            or values != tuple(sorted(values))
        ):
            raise ValueError("requested environment keys must be unique, sorted, and canonical")
        return values


class PopulationAssayResponseBlocker(_ScaffoldModel):
    code: PopulationAssayResponseBlockerCode
    detail: str = Field(min_length=1)

    @field_validator("detail")
    @classmethod
    def detail_is_canonical(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("blocker detail must be nonblank and trimmed")
        return value


class PopulationAssayResponsePreflight(_ScaffoldModel):
    """Derived request and bundle readiness; no readiness Boolean is caller supplied."""

    task_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    bundle_readiness: BundleReadiness
    scope_blockers: tuple[PopulationAssayResponseBlocker, ...] = ()
    readiness_blockers: tuple[PopulationAssayResponseBlocker, ...] = Field(min_length=1)

    @property
    def request_scope_supported(self) -> bool:
        return not self.scope_blockers

    @property
    def execution_allowed(self) -> bool:
        return (
            self.request_scope_supported
            and not self.readiness_blockers
            and self.bundle_readiness.runnable
        )

    @property
    def can_emit_population_response_distribution(self) -> bool:
        return self.execution_allowed

    @property
    def can_emit_cell_state_belief(self) -> Literal[False]:
        return False

    @model_validator(mode="after")
    def blockers_are_canonical(self) -> PopulationAssayResponsePreflight:
        for blockers, name in (
            (self.scope_blockers, "scope blockers"),
            (self.readiness_blockers, "readiness blockers"),
        ):
            keys = tuple((item.code.value, item.detail) for item in blockers)
            if len(keys) != len(set(keys)) or keys != tuple(sorted(keys)):
                raise ValueError(f"{name} must be unique and canonically sorted")
        if self.execution_allowed:
            raise ValueError("the component scaffold cannot become executable")
        return self


class _FrozenComponentInputs(_ScaffoldModel):
    manifest: DatasetManifest
    query: StateQuery
    benchmark: BenchmarkArtifact
    support_envelope: BiologicalSupportEnvelope
    bundle: BiologicalModelBundleContract


def _read_exact(path: Path, *, expected_sha256: str, name: str) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ContractViolationError(f"missing {name}: {path}") from error
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ContractViolationError(
            f"{name} SHA-256 drift: expected {expected_sha256}, observed {actual}"
        )
    return payload


def _read_canonical_model(
    path: Path,
    model_type: type[SchemaModel],
    *,
    expected_sha256: str,
    name: str,
) -> SchemaModel:
    try:
        payload = _read_exact(path, expected_sha256=expected_sha256, name=name)
        model = model_type.model_validate_json(payload)
    except (OSError, ValueError) as error:
        raise ContractViolationError(f"invalid {name}: {path}") from error
    if payload != canonical_json_bytes(model.model_dump(mode="json")):
        raise ContractViolationError(f"{name} is not canonical JSON: {path}")
    return model


def _load_frozen_inputs(repository_root: Path) -> _FrozenComponentInputs:
    manifest_payload = _read_exact(
        repository_root / _MANIFEST_PATH,
        expected_sha256=SCIPLEX3_K562_MANIFEST_SHA256,
        name="sci-Plex3 K562 manifest",
    )
    query_payload = _read_exact(
        repository_root / _QUERY_PATH,
        expected_sha256=SCIPLEX3_K562_QUERY_SHA256,
        name="sci-Plex3 K562 query",
    )
    benchmark_payload = _read_exact(
        repository_root / _BENCHMARK_PATH,
        expected_sha256=SCIPLEX3_K562_BENCHMARK_SHA256,
        name="sci-Plex3 K562 benchmark",
    )
    try:
        manifest = DatasetManifest.model_validate_json(manifest_payload)
        query = StateQuery.model_validate_json(query_payload)
        benchmark = BenchmarkArtifact.model_validate_json(benchmark_payload)
    except ValueError as error:
        raise ContractViolationError(
            "frozen component artifact failed schema validation"
        ) from error
    if manifest_payload != manifest.canonical_json_bytes:
        raise ContractViolationError("frozen manifest bytes are not canonical")
    if query_payload != canonical_json_bytes(query.model_dump(mode="json")):
        raise ContractViolationError("frozen query bytes are not canonical")
    if benchmark_payload != canonical_json_bytes(benchmark.model_dump(mode="json")):
        raise ContractViolationError("frozen benchmark bytes are not canonical")

    support_model = _read_canonical_model(
        repository_root / _SUPPORT_ENVELOPE_PATH,
        BiologicalSupportEnvelope,
        expected_sha256=SCIPLEX3_K562_SUPPORT_ENVELOPE_SHA256,
        name="population-response support envelope",
    )
    bundle_model = _read_canonical_model(
        repository_root / _BUNDLE_PATH,
        BiologicalModelBundleContract,
        expected_sha256=SCIPLEX3_K562_BUNDLE_CONTRACT_SHA256,
        name="population-response bundle contract",
    )
    assert isinstance(support_model, BiologicalSupportEnvelope)
    assert isinstance(bundle_model, BiologicalModelBundleContract)

    if benchmark.definition.query.state_query != query:
        raise ContractViolationError("benchmark does not embed the exact frozen StateQuery")
    if benchmark.definition.query.query_fingerprint != query.fingerprint:
        raise ContractViolationError("benchmark query fingerprint does not match frozen query")
    manifests = {binding.binding_id: manifest for binding in benchmark.definition.evidence_bindings}
    try:
        verify_benchmark_artifact(benchmark, manifests)
    except ValueError as error:
        raise ContractViolationError("frozen benchmark failed evidence re-verification") from error

    _validate_component_scope(query, benchmark)
    return _FrozenComponentInputs(
        manifest=manifest,
        query=query,
        benchmark=benchmark,
        support_envelope=support_model,
        bundle=bundle_model,
    )


def _validate_component_scope(query: StateQuery, benchmark: BenchmarkArtifact) -> None:
    target = query.target_outputs
    horizons = query.prediction_horizons
    if (
        query.system_boundary is not SystemBoundary.POPULATION
        or len(target) != 1
        or target[0].term.key != SCIPLEX3_K562_TARGET_KEY
        or target[0].aggregation.statistic.value != "distribution"
        or target[0].aggregation.experimental_unit.casefold() != "well"
        or len(horizons) != 1
        or horizons[0].name != SCIPLEX3_K562_HORIZON_NAME
        or horizons[0].duration_seconds != 86_400.0
        or query.environment_space
        or query.constraints.allow_transport
    ):
        raise ContractViolationError(
            "frozen query no longer has the exact endpoint component scope"
        )
    scope = benchmark.definition.scope
    if (
        scope.query_fingerprint != query.fingerprint
        or scope.forecast_causal_status is not CausalStatus.PREDICTIVE_ASSOCIATION
        or scope.target_output_keys != (SCIPLEX3_K562_TARGET_KEY,)
        or scope.horizon_names != (SCIPLEX3_K562_HORIZON_NAME,)
    ):
        raise ContractViolationError("benchmark scope no longer matches the endpoint component")


def _blocker(
    code: PopulationAssayResponseBlockerCode,
    detail: str,
) -> PopulationAssayResponseBlocker:
    return PopulationAssayResponseBlocker(code=code, detail=detail)


def _canonical_blockers(
    blockers: list[PopulationAssayResponseBlocker],
) -> tuple[PopulationAssayResponseBlocker, ...]:
    return tuple(sorted(set(blockers), key=lambda item: (item.code.value, item.detail)))


class SciPlex3K562PopulationAssayResponseScaffold:
    """Exact-scope gate for a future direct response model; never an inference backend."""

    def __init__(self, repository_root: Path) -> None:
        self._repository_root = repository_root.resolve()
        _load_frozen_inputs(self._repository_root)

    @classmethod
    def from_repository(
        cls,
        repository_root: Path,
    ) -> SciPlex3K562PopulationAssayResponseScaffold:
        return cls(repository_root)

    @property
    def component_fingerprint(self) -> str:
        return _load_frozen_inputs(self._repository_root).bundle.fingerprint

    def preflight(self, request: PopulationAssayResponseTask) -> PopulationAssayResponsePreflight:
        frozen = _load_frozen_inputs(self._repository_root)
        benchmark = frozen.benchmark
        manifests = {
            binding.binding_id: frozen.manifest
            for binding in benchmark.definition.evidence_bindings
        }
        readiness = assess_biological_model_bundle(
            frozen.bundle,
            query=frozen.query,
            benchmark=benchmark,
            manifests=manifests,
            support_envelope=frozen.support_envelope,
        )
        scope_blockers = self._scope_blockers(request, frozen)
        readiness_blockers = self._readiness_blockers(frozen, readiness)
        return PopulationAssayResponsePreflight(
            task_fingerprint=request.fingerprint,
            bundle_readiness=readiness,
            scope_blockers=_canonical_blockers(scope_blockers),
            readiness_blockers=_canonical_blockers(readiness_blockers),
        )

    def sample_response(
        self,
        request: PopulationAssayResponseTask,
        *,
        sample_count: int,
        seed: int,
    ) -> Never:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        preflight = self.preflight(request)
        details = tuple(
            blocker.detail for blocker in (*preflight.scope_blockers, *preflight.readiness_blockers)
        )
        raise BiologicalExecutionBlockedError("; ".join(details))

    @staticmethod
    def _scope_blockers(
        request: PopulationAssayResponseTask,
        frozen: _FrozenComponentInputs,
    ) -> list[PopulationAssayResponseBlocker]:
        blockers: list[PopulationAssayResponseBlocker] = []
        if request.query != frozen.query or request.query.fingerprint != SCIPLEX3_K562_QUERY_SHA256:
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.QUERY_MISMATCH,
                    "The request does not bind the exact frozen sci-Plex3 K562 StateQuery.",
                )
            )

        case_set = frozen.benchmark.definition.evaluation_case_set
        assert case_set is not None
        expected_by_id = {case.case_id: case for case in case_set.cases}
        expected = expected_by_id.get(request.evaluation_case.case_id)
        if expected is None:
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.EVALUATION_CASE_UNKNOWN,
                    "The requested well population is not an exact frozen evaluation case.",
                )
            )
        else:
            actual = request.evaluation_case
            if (actual.context_id, actual.context_fingerprint) != (
                expected.context_id,
                expected.context_fingerprint,
            ):
                blockers.append(
                    _blocker(
                        PopulationAssayResponseBlockerCode.EVALUATION_CASE_MISMATCH,
                        "The static plate context does not match the exact frozen well case.",
                    )
                )
            if actual.intervention_spec_ids != expected.intervention_spec_ids:
                blockers.append(
                    _blocker(
                        PopulationAssayResponseBlockerCode.ACTION_OR_DOSE_MISMATCH,
                        "The assigned action/no-action or exact dose differs from the frozen case.",
                    )
                )
            if actual.horizon_name != expected.horizon_name:
                blockers.append(
                    _blocker(
                        PopulationAssayResponseBlockerCode.HORIZON_MISMATCH,
                        "Only the exact 24h-endpoint horizon is in component scope.",
                    )
                )
            if actual.target_output_keys != expected.target_output_keys:
                blockers.append(
                    _blocker(
                        PopulationAssayResponseBlockerCode.TARGET_MISMATCH,
                        "Only the exact recovered-nucleus 2,000-feature UMI distribution is in "
                        "scope.",
                    )
                )
            compared = actual.model_copy(
                update={
                    "context_id": expected.context_id,
                    "context_fingerprint": expected.context_fingerprint,
                    "intervention_spec_ids": expected.intervention_spec_ids,
                    "horizon_name": expected.horizon_name,
                    "target_output_keys": expected.target_output_keys,
                }
            )
            if compared != expected:
                blockers.append(
                    _blocker(
                        PopulationAssayResponseBlockerCode.EVALUATION_CASE_MISMATCH,
                        "The subject, partition, controls, or matching stratum differs from the "
                        "frozen case.",
                    )
                )

        partitions = frozen.benchmark.definition.split_plan
        assert partitions is not None
        role_by_id = {partition.partition_id: partition.role for partition in partitions.partitions}
        actual_role = role_by_id.get(request.evaluation_case.partition_id)
        expected_access = {
            PopulationComponentAccessPurpose.TRAIN_PARAMETERS: (
                "p1-train",
                BenchmarkPartitionRole.TRAIN,
            ),
            PopulationComponentAccessPurpose.FIT_CALIBRATION: (
                "p2-calibration",
                BenchmarkPartitionRole.CALIBRATION,
            ),
            PopulationComponentAccessPurpose.MODEL_SELECTION: (
                "p3-model-selection-validation",
                BenchmarkPartitionRole.MODEL_SELECTION_VALIDATION,
            ),
            PopulationComponentAccessPurpose.UNTOUCHED_EVALUATION: (
                "p4-untouched-test",
                BenchmarkPartitionRole.UNTOUCHED_TEST,
            ),
        }
        expected_partition_id, expected_role = expected_access[request.access_purpose]
        if (
            request.evaluation_case.partition_id != expected_partition_id
            or actual_role is not expected_role
        ):
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.PARTITION_ROLE_MISMATCH,
                    "Requested lifecycle access does not match its frozen partition and role.",
                )
            )
        if (
            request.access_purpose is PopulationComponentAccessPurpose.TRAIN_PARAMETERS
            and actual_role is not BenchmarkPartitionRole.TRAIN
        ):
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.TRAINING_PARTITION_VIOLATION,
                    "Parameter training is restricted to p1-train.",
                )
            )
        if (
            request.access_purpose is PopulationComponentAccessPurpose.TRAIN_PARAMETERS
            and actual_role is BenchmarkPartitionRole.UNTOUCHED_TEST
        ):
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.UNTOUCHED_TEST_MUTATION,
                    "p4-untouched-test is locked against tuning and parameter updates.",
                )
            )

        allowed_pre_cutoff = {term.key for term in frozen.query.evidence_policy.allowed_modalities}
        supplied_pre_cutoff = {term.key for term in request.pre_cutoff_evidence_modalities}
        if SCIPLEX3_K562_ENDPOINT_MODALITY_KEY in supplied_pre_cutoff:
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.ENDPOINT_TARGET_LEAKAGE,
                    "The future single-nucleus RNA endpoint cannot be used before the t=0 cutoff.",
                )
            )
        if not supplied_pre_cutoff <= allowed_pre_cutoff:
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.PRE_CUTOFF_EVIDENCE_UNSUPPORTED,
                    "Pre-cutoff inputs are limited to the frozen design-context modality.",
                )
            )
        if request.output_representation is not PopulationResponseRepresentation.PREDICTIVE_SAMPLES:
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.POINT_ONLY_OUTPUT_UNSUPPORTED,
                    "A point response cannot substitute for a predictive population distribution.",
                )
            )
        if request.forecast_causal_status is not CausalStatus.PREDICTIVE_ASSOCIATION:
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.CAUSAL_STATUS_OVERCLAIM,
                    "Component forecasts are predictive associations, not identified or "
                    "transported effects.",
                )
            )
        if request.requested_environment_keys:
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.ENVIRONMENT_UNSUPPORTED,
                    "The frozen query contains no environment intervention domain.",
                )
            )
        if request.transport_requested:
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.TRANSPORT_UNSUPPORTED,
                    "No cross-study, donor, cell-line, or environment transport is in scope.",
                )
            )
        if request.identified_intervention_realization_required:
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.INTERVENTION_REALIZATION_UNIDENTIFIED,
                    "Intracellular exposure and target engagement remain unknown.",
                )
            )
        if request.survival_or_viability_interpretation_required:
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.SURVIVAL_VIABILITY_UNSUPPORTED,
                    "Recovered-nucleus RNA is not a viability, survival, or censoring model.",
                )
            )
        if request.current_hidden_state_inference_required:
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.CURRENT_HIDDEN_STATE_UNSUPPORTED,
                    "Static design context is not an inferred t=0 hidden cell state.",
                )
            )
        return blockers

    @staticmethod
    def _readiness_blockers(
        frozen: _FrozenComponentInputs,
        readiness: BundleReadiness,
    ) -> list[PopulationAssayResponseBlocker]:
        benchmark = frozen.benchmark
        blockers = [
            _blocker(
                PopulationAssayResponseBlockerCode.BUNDLE_NOT_RUNNABLE,
                "The checked-in component bundle is a non-runnable scaffold.",
            ),
            _blocker(
                PopulationAssayResponseBlockerCode.RUNTIME_SUPPORT_NOT_ADMITTED,
                "No biological runtime support envelope has passed training and validation gates.",
            ),
        ]
        if frozen.bundle.model_artifact is None:
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.MODEL_ARTIFACT_ABSENT,
                    "Trained population-response weights and their training run are absent.",
                )
            )
        if not frozen.bundle.validation_evidence:
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.VALIDATION_EVIDENCE_ABSENT,
                    "Calibration, support/OOD, model-selection, and locked-test evidence are "
                    "absent.",
                )
            )
        if any(
            isinstance(metric.implementation_binding, SpecificationOnlyImplementationBinding)
            or isinstance(metric.uncertainty.method, SpecificationOnlyImplementationBinding)
            for metric in benchmark.definition.metrics
        ):
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.METRIC_IMPLEMENTATIONS_ABSENT,
                    "Benchmark metrics or their uncertainty procedures remain specification-only.",
                )
            )
        applicable_runs = tuple(
            run
            for baseline, run in zip(
                benchmark.definition.baselines,
                benchmark.admission.baseline_runs,
                strict=True,
            )
            if baseline.applicability.applies_to(frozen.query)
        )
        if not applicable_runs or any(
            run.status is not BaselineRunStatus.PASSED for run in applicable_runs
        ):
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.BASELINES_NOT_EXECUTED,
                    "Applicable benchmark baselines have not passed on frozen cases.",
                )
            )
        source_duplicate_checks = tuple(
            check
            for check in (() if benchmark.leakage_audit is None else benchmark.leakage_audit.checks)
            if check.kind is LeakageCheckKind.SOURCE_DUPLICATE_DISJOINT
        )
        if not source_duplicate_checks or any(
            check.status is not AuditCheckStatus.PASSED for check in source_duplicate_checks
        ):
            blockers.append(
                _blocker(
                    PopulationAssayResponseBlockerCode.SOURCE_DUPLICATE_AUDIT_UNASSESSED,
                    "Source-duplicate leakage remains unassessed beyond exact source record "
                    "identity.",
                )
            )
        if readiness.runnable:
            raise ContractViolationError("a component scaffold was incorrectly assessed runnable")
        return blockers


__all__ = [
    "SCIPLEX3_K562_BENCHMARK_SHA256",
    "SCIPLEX3_K562_BUNDLE_CONTRACT_SHA256",
    "SCIPLEX3_K562_FEATURE_COUNT",
    "SCIPLEX3_K562_HORIZON_NAME",
    "SCIPLEX3_K562_MANIFEST_SHA256",
    "SCIPLEX3_K562_ORDERED_FEATURE_KEYS_SHA256",
    "SCIPLEX3_K562_QUERY_SHA256",
    "SCIPLEX3_K562_SUPPORT_ENVELOPE_SHA256",
    "SCIPLEX3_K562_TARGET_KEY",
    "SCIPLEX3_K562_TARGET_VALUE_SCHEMA_SHA256",
    "PopulationAssayResponseBlocker",
    "PopulationAssayResponseBlockerCode",
    "PopulationAssayResponsePreflight",
    "PopulationAssayResponseTask",
    "PopulationComponentAccessPurpose",
    "PopulationResponseRepresentation",
    "SciPlex3K562PopulationAssayResponseScaffold",
]
