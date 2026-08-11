"""Content-addressed biological bundle and support-port admission contracts.

The models in this module are deliberately able to describe an incomplete component scaffold.
They never accept a caller-supplied ``runnable`` or ``admitted`` flag.  Those states are derived by
``assess_biological_model_bundle`` after re-binding the exact query, benchmark, support envelope,
training run, validation evidence, model artifact, and every stage port.

This contract is additive to :class:`cellstate.ports.CellStateModelBundle`.  An admitted runtime
backend can continue implementing that protocol; an experimental component cannot acquire a
biological ``EstimatorDescriptor`` merely by satisfying Python interfaces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from cellstate.data.benchmarks import (
    AuditCheckStatus,
    BaselineRunStatus,
    BenchmarkArtifact,
    BenchmarkPartitionRole,
    ContentAddressedArtifact,
    ExecutableImplementationBinding,
    verify_benchmark_artifact,
)
from cellstate.data.manifests import DatasetManifest
from cellstate.domain.common import SchemaModel, canonical_fingerprint, canonical_json_bytes
from cellstate.domain.events import EvidenceRole
from cellstate.domain.query import (
    AssayPurpose,
    FutureAssayObservationEndpoint,
    StateQuery,
    SystemBoundary,
)
from cellstate.domain.subjects import SubjectKind
from cellstate.ports.models import EstimatorDescriptor, ModelArtifactKind

if TYPE_CHECKING:
    from .training import TrainingVerificationContext
    from .validation import (
        AdmissionVerificationContext,
        BiologicalExecutionAuthorization,
        JITCodeProvider,
    )

BundleContractSchemaVersion = Literal["0.1-experimental"]
BUNDLE_CONTRACT_SCHEMA_VERSION: BundleContractSchemaVersion = "0.1-experimental"
QUERY_PREREQUISITE_COMPILER_ID = "cellstate.query-derived-port-prerequisite-compiler"
QUERY_PREREQUISITE_COMPILER_VERSION = "0.1.0"


class BundleContractModel(SchemaModel):
    """Strict base used for experimental backend contracts."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
        strict=True,
    )


def _canonical_text(value: str, *, name: str) -> str:
    if not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be nonblank and trimmed")
    return value


def _canonical_values(
    values: tuple[str, ...],
    *,
    name: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not values and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if any(not value.strip() or value != value.strip() for value in values):
        raise ValueError(f"{name} must contain canonical nonblank strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")
    if values != tuple(sorted(values)):
        raise ValueError(f"{name} must be sorted")
    return values


class BundleContractKind(StrEnum):
    """Scientific scope of an artifact, not a readiness or admission assertion."""

    COMPONENT_SCAFFOLD = "component_scaffold"
    COMPONENT_MODEL = "component_model"
    BIOLOGICAL_MODEL_BUNDLE = "biological_model_bundle"


class ComponentLifecycleStage(StrEnum):
    """Derived component progress; stages are never declared on a source bundle."""

    SCAFFOLD = "scaffold"
    TRAINED_CANDIDATE = "trained_candidate"
    CALIBRATED_CANDIDATE = "calibrated_candidate"
    MODEL_SELECTED_FROZEN = "model_selected_frozen"
    COMPONENT_EVALUATED = "component_evaluated"
    COMPONENT_GATES_PASSED = "component_gates_passed"


class BundleAdmissionBlockerCode(StrEnum):
    """Machine-readable gates derived by the trusted admission boundary."""

    ARTIFACT_BYTES_UNRESOLVED = "artifact_bytes_unresolved"
    IMPLEMENTATION_INTERFACES_UNVERIFIED = "implementation_interfaces_unverified"
    QUERY_DERIVED_OPERATION_PREREQUISITES_UNVERIFIED = (
        "query_derived_operation_prerequisites_unverified"
    )
    VALIDATION_RESULTS_FAILED = "validation_results_failed"
    VALIDATION_RESULTS_UNVERIFIED = "validation_results_unverified"


class DirectPopulationResponseSemantics(BundleContractModel):
    """Fail-closed meaning of the endpoint-only component computation.

    Literal fields prevent a future endpoint or its matched vehicle observations from being
    relabeled as a pre-cutoff prior merely to satisfy a cell-state estimator interface.
    """

    computation: Literal["direct_population_assay_response"] = "direct_population_assay_response"
    inference_inputs: Literal["static_context_and_assigned_action_only"] = (
        "static_context_and_assigned_action_only"
    )
    endpoint_observations: Literal["future_target_only"] = "future_target_only"
    matched_control_observations: Literal["endpoint_comparator_only"] = "endpoint_comparator_only"


class ModelOperation(StrEnum):
    """Public runtime operation a support envelope may eventually register."""

    CHOOSE_INTERVENTION = "choose_intervention"
    ESTIMATE_CELL_STATE = "estimate_cell_state"
    EVOLVE_CELL_STATE = "evolve_cell_state"
    RECOMMEND_NEXT_MEASUREMENT = "recommend_next_measurement"


class BiologicalStagePort(StrEnum):
    """Original framework stages plus the narrow component's support infrastructure.

    The first eighteen named biological stages are the complete port surface from the original
    proposed ``CellStateModelBundle`` skeleton.  The remaining support ports let a component be
    implemented and benchmarked without pretending that it is a current-state belief backend.
    """

    # Original proposed framework stages.
    QUERY_COMPILER = "query_compiler"
    REFERENCE_PRIOR = "reference_prior"
    OBSERVATION_MODELS = "observation_models"
    EVIDENCE_TRANSFER_MODELS = "evidence_transfer_models"
    INTERVENTION_REALIZATION_MODEL = "intervention_realization_model"
    TRANSITION_MODEL = "transition_model"
    DIVISION_AND_INHERITANCE_MODEL = "division_and_inheritance_model"
    CELL_INTERACTION_MODEL = "cell_interaction_model"
    EXTRACELLULAR_TRANSPORT_MODEL = "extracellular_transport_model"
    FUNCTIONAL_DECODERS = "functional_decoders"
    MECHANISTIC_CONSTRAINTS = "mechanistic_constraints"
    POSTERIOR_INFERENCE_ENGINE = "posterior_inference_engine"
    MODEL_ENSEMBLE = "model_ensemble"
    UNCERTAINTY_CALIBRATOR = "uncertainty_calibrator"
    OOD_DETECTOR = "ood_detector"
    SUFFICIENCY_EVALUATOR = "sufficiency_evaluator"
    IDENTIFIABILITY_ANALYZER = "identifiability_analyzer"
    VALUE_OF_INFORMATION_ENGINE = "value_of_information_engine"

    # Infrastructure required to build the first narrow population-response component honestly.
    EXACT_ARTIFACT_RESOLVER = "exact_artifact_resolver"
    QUERY_SCOPE_VALIDATOR = "query_scope_validator"
    TRAIN_CAL_DATA_LOADER = "train_cal_data_loader"
    ACTION_CONTEXT_ENCODER = "action_context_encoder"
    POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL = "population_assay_response_distribution_model"
    STRICT_SUPPORT_OOD_GATE = "strict_support_ood_gate"
    ARTIFACT_PROVENANCE_WRITER = "artifact_provenance_writer"


RUNTIME_FOUNDATION_PORTS = frozenset(
    {
        BiologicalStagePort.ARTIFACT_PROVENANCE_WRITER,
        BiologicalStagePort.EXACT_ARTIFACT_RESOLVER,
        BiologicalStagePort.QUERY_SCOPE_VALIDATOR,
        BiologicalStagePort.STRICT_SUPPORT_OOD_GATE,
    }
)


OPERATION_REQUIRED_PORTS: Mapping[ModelOperation, frozenset[BiologicalStagePort]] = (
    MappingProxyType(
        {
            ModelOperation.ESTIMATE_CELL_STATE: RUNTIME_FOUNDATION_PORTS
            | frozenset(
                {
                    BiologicalStagePort.IDENTIFIABILITY_ANALYZER,
                    BiologicalStagePort.OOD_DETECTOR,
                    BiologicalStagePort.POSTERIOR_INFERENCE_ENGINE,
                    BiologicalStagePort.QUERY_COMPILER,
                    BiologicalStagePort.REFERENCE_PRIOR,
                    BiologicalStagePort.SUFFICIENCY_EVALUATOR,
                    BiologicalStagePort.UNCERTAINTY_CALIBRATOR,
                }
            ),
            ModelOperation.EVOLVE_CELL_STATE: RUNTIME_FOUNDATION_PORTS
            | frozenset(
                {
                    BiologicalStagePort.FUNCTIONAL_DECODERS,
                    BiologicalStagePort.OOD_DETECTOR,
                    BiologicalStagePort.TRANSITION_MODEL,
                    BiologicalStagePort.UNCERTAINTY_CALIBRATOR,
                }
            ),
            ModelOperation.CHOOSE_INTERVENTION: RUNTIME_FOUNDATION_PORTS
            | frozenset(
                {
                    BiologicalStagePort.FUNCTIONAL_DECODERS,
                    BiologicalStagePort.OOD_DETECTOR,
                    BiologicalStagePort.TRANSITION_MODEL,
                    BiologicalStagePort.UNCERTAINTY_CALIBRATOR,
                }
            ),
            ModelOperation.RECOMMEND_NEXT_MEASUREMENT: RUNTIME_FOUNDATION_PORTS
            | frozenset(
                {
                    BiologicalStagePort.FUNCTIONAL_DECODERS,
                    BiologicalStagePort.OBSERVATION_MODELS,
                    BiologicalStagePort.OOD_DETECTOR,
                    BiologicalStagePort.POSTERIOR_INFERENCE_ENGINE,
                    BiologicalStagePort.TRANSITION_MODEL,
                    BiologicalStagePort.UNCERTAINTY_CALIBRATOR,
                    BiologicalStagePort.VALUE_OF_INFORMATION_ENGINE,
                }
            ),
        }
    )
)


ORIGINAL_SKELETON_PORTS = frozenset(
    {
        BiologicalStagePort.QUERY_COMPILER,
        BiologicalStagePort.REFERENCE_PRIOR,
        BiologicalStagePort.OBSERVATION_MODELS,
        BiologicalStagePort.EVIDENCE_TRANSFER_MODELS,
        BiologicalStagePort.INTERVENTION_REALIZATION_MODEL,
        BiologicalStagePort.TRANSITION_MODEL,
        BiologicalStagePort.DIVISION_AND_INHERITANCE_MODEL,
        BiologicalStagePort.CELL_INTERACTION_MODEL,
        BiologicalStagePort.EXTRACELLULAR_TRANSPORT_MODEL,
        BiologicalStagePort.FUNCTIONAL_DECODERS,
        BiologicalStagePort.MECHANISTIC_CONSTRAINTS,
        BiologicalStagePort.POSTERIOR_INFERENCE_ENGINE,
        BiologicalStagePort.MODEL_ENSEMBLE,
        BiologicalStagePort.UNCERTAINTY_CALIBRATOR,
        BiologicalStagePort.OOD_DETECTOR,
        BiologicalStagePort.SUFFICIENCY_EVALUATOR,
        BiologicalStagePort.IDENTIFIABILITY_ANALYZER,
        BiologicalStagePort.VALUE_OF_INFORMATION_ENGINE,
    }
)


class PortDisposition(StrEnum):
    """Honest implementation state for one explicitly mapped port."""

    REQUIRED = "required"
    PROVIDED = "provided"
    PLANNED = "planned"
    NOT_APPLICABLE = "not_applicable"
    UNSUPPORTED = "unsupported"


class QueryPrerequisiteTargetKind(StrEnum):
    """Exact computation surface for one query-derived prerequisite set."""

    DIRECT_POPULATION_RESPONSE_COMPONENT = "direct_population_response_component"
    RUNTIME_OPERATION = "runtime_operation"


class QueryPrerequisiteReasonCode(StrEnum):
    """Machine-readable reason why one biological stage is mandatory."""

    ACTION_CONTEXT_ENCODING = "action_context_encoding"
    CAUSAL_TRANSPORT_ASSUMPTIONS = "causal_transport_assumptions"
    COMPONENT_ARTIFACT_IDENTITY = "component_artifact_identity"
    COMPONENT_PARTITION_LOADING = "component_partition_loading"
    COMPONENT_PROVENANCE = "component_provenance"
    COMPONENT_QUERY_SCOPE = "component_query_scope"
    COMPONENT_RESPONSE_DISTRIBUTION = "component_response_distribution"
    COMPONENT_SUPPORT_OOD = "component_support_ood"
    COMPONENT_UNCERTAINTY_CALIBRATION = "component_uncertainty_calibration"
    COUNTERFACTUAL_REPLANNING = "counterfactual_replanning"
    DECLARED_MECHANISTIC_SEMANTICS = "declared_mechanistic_semantics"
    ENVIRONMENT_HISTORY_DYNAMICS = "environment_history_dynamics"
    EPISTEMIC_MODEL_UNCERTAINTY = "epistemic_model_uncertainty"
    FUTURE_ASSAY_TARGET = "future_assay_target"
    FUNCTIONAL_TARGET = "functional_target"
    HYPOTHETICAL_MEASUREMENT_UPDATE = "hypothetical_measurement_update"
    INTERVENTION_HISTORY_DYNAMICS = "intervention_history_dynamics"
    INTERVENTION_REALIZATION_UNCERTAINTY = "intervention_realization_uncertainty"
    LINEAGE_OR_POPULATION_DYNAMICS = "lineage_or_population_dynamics"
    MEASUREMENT_DECISION_EVSI = "measurement_decision_evsi"
    MEASUREMENT_OUTCOME_PREDICTION = "measurement_outcome_prediction"
    NEIGHBORHOOD_OR_POPULATION_INTERACTIONS = "neighborhood_or_population_interactions"
    NONDIRECT_EVIDENCE_TRANSFER = "nondirect_evidence_transfer"
    OBSERVATION_LIKELIHOOD = "observation_likelihood"
    OPERATION_CONTRACT_FLOOR = "operation_contract_floor"
    PLANNING_IDENTIFIABILITY = "planning_identifiability"
    PLANNING_STATE_SUFFICIENCY = "planning_state_sufficiency"
    SOLUBLE_OR_SPATIAL_TRANSPORT = "soluble_or_spatial_transport"
    TEMPORAL_EVIDENCE_PROPAGATION = "temporal_evidence_propagation"


class QueryPrerequisiteScopeIssue(StrEnum):
    """Query/target combinations that cannot define the requested computation."""

    INTERVENTION_PLANNING_REQUIRES_ACTION_SPACE = "intervention_planning_requires_action_space"
    MEASUREMENT_EVSI_REQUIRES_ACTION_SPACE = "measurement_evsi_requires_action_space"
    MEASUREMENT_SELECTION_REQUIRES_CANDIDATE_ASSAYS = (
        "measurement_selection_requires_candidate_assays"
    )
    RUNTIME_BUNDLE_REQUIRES_OPERATION = "runtime_bundle_requires_operation"


class QueryDerivedPortPrerequisite(BundleContractModel):
    """One required port and every query-derived reason that requires it."""

    port: BiologicalStagePort
    reasons: tuple[QueryPrerequisiteReasonCode, ...] = Field(min_length=1)

    @field_validator("reasons")
    @classmethod
    def reasons_are_canonical(
        cls,
        values: tuple[QueryPrerequisiteReasonCode, ...],
    ) -> tuple[QueryPrerequisiteReasonCode, ...]:
        reason_values = tuple(reason.value for reason in values)
        _canonical_values(
            reason_values,
            name="query prerequisite reason codes",
            allow_empty=False,
        )
        return values


class QueryDerivedPrerequisiteTarget(BundleContractModel):
    """Prerequisites for one exact public operation or direct component surface."""

    target_kind: QueryPrerequisiteTargetKind
    operation: ModelOperation | None = None
    direct_population_response: DirectPopulationResponseSemantics | None = None
    required_ports: tuple[QueryDerivedPortPrerequisite, ...] = Field(min_length=1)

    @property
    def target_id(self) -> str:
        if self.operation is not None:
            return f"runtime_operation:{self.operation.value}"
        return "component:direct_population_assay_response"

    @model_validator(mode="after")
    def target_and_ports_are_canonical(self) -> QueryDerivedPrerequisiteTarget:
        is_runtime = self.target_kind is QueryPrerequisiteTargetKind.RUNTIME_OPERATION
        if is_runtime is (self.operation is None):
            raise ValueError("only a runtime prerequisite target names an operation")
        if is_runtime is not (self.direct_population_response is None):
            raise ValueError(
                "only a direct-component prerequisite target binds component semantics"
            )
        port_values = tuple(item.port.value for item in self.required_ports)
        _canonical_values(
            port_values,
            name="query-derived target prerequisite ports",
            allow_empty=False,
        )
        return self


class QueryPrerequisiteDispositionFailure(BundleContractModel):
    """A derived-required port classified as unavailable by the bundle."""

    port: BiologicalStagePort
    disposition: PortDisposition

    @model_validator(mode="after")
    def disposition_is_a_failure(self) -> QueryPrerequisiteDispositionFailure:
        if self.disposition not in {
            PortDisposition.PLANNED,
            PortDisposition.NOT_APPLICABLE,
            PortDisposition.UNSUPPORTED,
        }:
            raise ValueError(
                "a query-prerequisite disposition failure must be planned, not applicable, "
                "or unsupported"
            )
        return self


class QueryDerivedPrerequisiteReport(BundleContractModel):
    """Deterministic query/envelope/bundle-bound prerequisite compilation.

    The report is evidence to be *bound* by a future trusted admission receipt, never an
    authorization by itself. Verification must call :func:`verify_query_prerequisite_report`,
    which re-derives this payload from the exact source objects.
    """

    compiler_id: str = Field(min_length=1)
    compiler_version: str = Field(min_length=1)
    compiler_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    support_envelope_id: str = Field(min_length=1)
    support_envelope_version: str = Field(min_length=1)
    support_envelope_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    bundle_id: str = Field(min_length=1)
    bundle_version: str = Field(min_length=1)
    bundle_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    bundle_kind: BundleContractKind
    targets: tuple[QueryDerivedPrerequisiteTarget, ...]
    required_ports: tuple[BiologicalStagePort, ...]
    envelope_missing_ports: tuple[BiologicalStagePort, ...]
    envelope_extra_ports: tuple[BiologicalStagePort, ...]
    invalid_dispositions: tuple[QueryPrerequisiteDispositionFailure, ...]
    scope_issues: tuple[QueryPrerequisiteScopeIssue, ...]

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))

    @property
    def structurally_satisfied(self) -> bool:
        """Whether declared scope/dispositions satisfy derivation, absent trusted receipts."""

        return not any(
            (
                self.envelope_missing_ports,
                self.envelope_extra_ports,
                self.invalid_dispositions,
                self.scope_issues,
            )
        )

    @field_validator(
        "compiler_fingerprint",
        "query_fingerprint",
        "support_envelope_fingerprint",
        "bundle_fingerprint",
    )
    @classmethod
    def fingerprints_are_lowercase(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def report_is_canonical(self) -> QueryDerivedPrerequisiteReport:
        if (
            self.compiler_id != QUERY_PREREQUISITE_COMPILER_ID
            or self.compiler_version != QUERY_PREREQUISITE_COMPILER_VERSION
            or self.compiler_fingerprint != QUERY_PREREQUISITE_COMPILER_FINGERPRINT
        ):
            raise ValueError("query prerequisite report must bind the canonical compiler identity")
        for value, name in (
            (self.compiler_id, "prerequisite compiler ID"),
            (self.compiler_version, "prerequisite compiler version"),
            (self.support_envelope_id, "prerequisite support-envelope ID"),
            (self.support_envelope_version, "prerequisite support-envelope version"),
            (self.bundle_id, "prerequisite bundle ID"),
            (self.bundle_version, "prerequisite bundle version"),
        ):
            _canonical_text(value, name=name)
        target_ids = tuple(target.target_id for target in self.targets)
        _canonical_values(target_ids, name="query prerequisite target IDs")
        required_values = tuple(port.value for port in self.required_ports)
        _canonical_values(required_values, name="query-derived required ports")
        target_ports = {
            prerequisite.port for target in self.targets for prerequisite in target.required_ports
        }
        if set(self.required_ports) != target_ports:
            raise ValueError(
                "query-derived required-port union must equal the per-target prerequisites"
            )
        for ports, name in (
            (self.envelope_missing_ports, "query prerequisite envelope-missing ports"),
            (self.envelope_extra_ports, "query prerequisite envelope-extra ports"),
        ):
            values = tuple(port.value for port in ports)
            _canonical_values(values, name=name)
        failure_values = tuple(
            f"{failure.port.value}:{failure.disposition.value}"
            for failure in self.invalid_dispositions
        )
        _canonical_values(
            failure_values,
            name="query prerequisite invalid dispositions",
        )
        issue_values = tuple(issue.value for issue in self.scope_issues)
        _canonical_values(issue_values, name="query prerequisite scope issues")
        return self


class PortImplementationKind(StrEnum):
    """Structural declaration kind; neither value is executable evidence."""

    SPECIFICATION_ONLY = "specification_only"
    PYTHON_ENTRY_POINT = "python_entry_point"


class BundleContractReference(BundleContractModel):
    """Content identity for one canonical contract payload."""

    contract_id: str = Field(min_length=1)
    contract_version: str = Field(min_length=1)
    artifact: ContentAddressedArtifact

    @model_validator(mode="after")
    def identity_is_canonical(self) -> BundleContractReference:
        _canonical_text(self.contract_id, name="contract ID")
        _canonical_text(self.contract_version, name="contract version")
        return self


class PortImplementationBinding(BundleContractModel):
    """Versioned code declaration for one port, not proof that its bytes can execute."""

    implementation_id: str = Field(min_length=1)
    implementation_version: str = Field(min_length=1)
    interface: str = Field(min_length=1)
    kind: PortImplementationKind
    code_artifact: ContentAddressedArtifact
    entrypoint: str | None = Field(default=None, min_length=1)

    @property
    def executable(self) -> bool:
        """Remain fail closed until a future byte resolver and interface verifier exist."""

        return False

    @property
    def declares_python_entry_point(self) -> bool:
        """Return whether the source contract names an entry point, without authorizing it."""

        return self.kind is PortImplementationKind.PYTHON_ENTRY_POINT

    @model_validator(mode="after")
    def executable_binding_has_entrypoint(self) -> PortImplementationBinding:
        _canonical_text(self.implementation_id, name="implementation ID")
        _canonical_text(self.implementation_version, name="implementation version")
        _canonical_text(self.interface, name="implementation interface")
        if self.entrypoint is not None:
            _canonical_text(self.entrypoint, name="implementation entrypoint")
        if (self.kind is PortImplementationKind.PYTHON_ENTRY_POINT) is (self.entrypoint is None):
            raise ValueError("only an executable Python implementation declares an entrypoint")
        return self


class ModelOperationImplementationBinding(BundleContractModel):
    """Exact high-level runtime declaration; stage ports alone cannot register an API."""

    operation: ModelOperation
    implementation: PortImplementationBinding
    validation_evidence_ids: tuple[str, ...] = ()
    rationale: tuple[str, ...] = Field(min_length=1)

    @field_validator("validation_evidence_ids")
    @classmethod
    def evidence_ids_are_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_values(values, name="operation validation evidence IDs")

    @field_validator("rationale")
    @classmethod
    def rationale_is_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_values(values, name="operation rationale", allow_empty=False)


class ModelPortBinding(BundleContractModel):
    """One complete, evidence-aware entry in the bundle support-port map."""

    port: BiologicalStagePort
    disposition: PortDisposition
    implementation: PortImplementationBinding | None = None
    validation_evidence_ids: tuple[str, ...] = ()
    rationale: tuple[str, ...] = Field(min_length=1)

    @field_validator("validation_evidence_ids")
    @classmethod
    def evidence_ids_are_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_values(values, name="port validation evidence IDs")

    @field_validator("rationale")
    @classmethod
    def rationale_is_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_values(values, name="port rationale", allow_empty=False)

    @model_validator(mode="after")
    def implementation_matches_disposition(self) -> ModelPortBinding:
        if self.disposition is PortDisposition.PROVIDED:
            if self.implementation is None:
                raise ValueError("a provided port requires an implementation binding")
        elif self.implementation is not None:
            raise ValueError("only a provided port may bind an implementation")
        if self.disposition is not PortDisposition.PROVIDED and self.validation_evidence_ids:
            raise ValueError("only a provided port may cite validation evidence")
        return self


class ValidationEvidenceKind(StrEnum):
    """Scientific role of a validation result; artifact names never imply this role."""

    LOCKED_COMPONENT_EVALUATION = "locked_component_evaluation"
    MODEL_SELECTION = "model_selection"
    PORT_IMPLEMENTATION_VALIDATION = "port_implementation_validation"
    RUNTIME_OPERATION_VALIDATION = "runtime_operation_validation"
    SUPPORT_OOD_VALIDATION = "support_ood_validation"
    UNCERTAINTY_CALIBRATION = "uncertainty_calibration"


class ValidationEvidenceRequirement(BundleContractModel):
    """Typed evidence role and exact benchmark partition roles required by an envelope."""

    evidence_id: str = Field(min_length=1)
    evidence_kind: ValidationEvidenceKind
    partition_roles: tuple[BenchmarkPartitionRole, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def requirement_is_canonical(self) -> ValidationEvidenceRequirement:
        _canonical_text(self.evidence_id, name="validation-evidence requirement ID")
        role_values = tuple(role.value for role in self.partition_roles)
        _canonical_values(
            role_values,
            name="validation-evidence requirement partition roles",
            allow_empty=False,
        )
        if BenchmarkPartitionRole.TRAIN in self.partition_roles:
            raise ValueError("validation evidence cannot use the training partition role")
        return self


class BiologicalSupportEnvelope(BundleContractModel):
    """Exact scientific and runtime scope against which a bundle is assessed."""

    schema_version: BundleContractSchemaVersion = BUNDLE_CONTRACT_SCHEMA_VERSION
    envelope_id: str = Field(min_length=1)
    envelope_version: str = Field(min_length=1)
    bundle_kind: BundleContractKind
    query: BundleContractReference
    benchmark: BundleContractReference
    direct_population_response: DirectPopulationResponseSemantics | None = None
    runtime_operations: tuple[ModelOperation, ...] = ()
    required_ports: tuple[BiologicalStagePort, ...] = Field(min_length=1)
    required_validation_evidence: tuple[ValidationEvidenceRequirement, ...] = ()
    notes: tuple[str, ...] = Field(min_length=1)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))

    @property
    def required_validation_evidence_ids(self) -> tuple[str, ...]:
        """Canonical IDs retained as a read-only compatibility view."""

        return tuple(item.evidence_id for item in self.required_validation_evidence)

    @model_validator(mode="after")
    def scope_is_canonical(self) -> BiologicalSupportEnvelope:
        _canonical_text(self.envelope_id, name="support-envelope ID")
        _canonical_text(self.envelope_version, name="support-envelope version")
        operation_values = tuple(operation.value for operation in self.runtime_operations)
        _canonical_values(operation_values, name="runtime operations")
        port_values = tuple(port.value for port in self.required_ports)
        _canonical_values(port_values, name="required support ports", allow_empty=False)
        evidence_ids = tuple(item.evidence_id for item in self.required_validation_evidence)
        _canonical_values(evidence_ids, name="required validation evidence IDs")
        _canonical_values(self.notes, name="support-envelope notes", allow_empty=False)
        is_component = self.bundle_kind in {
            BundleContractKind.COMPONENT_SCAFFOLD,
            BundleContractKind.COMPONENT_MODEL,
        }
        if is_component and self.runtime_operations:
            raise ValueError("a direct component cannot register public cell-state operations")
        if is_component is (self.direct_population_response is None):
            raise ValueError("only a direct component requires population-response semantics")
        if self.runtime_operations and not self.required_validation_evidence:
            raise ValueError("a runtime support envelope requires named validation evidence")
        required = set(self.required_ports)
        for operation in self.runtime_operations:
            missing = OPERATION_REQUIRED_PORTS[operation] - required
            if missing:
                names = ", ".join(sorted(port.value for port in missing))
                raise ValueError(
                    f"{operation.value} support envelope omits prerequisite ports: {names}"
                )
        return self


class TrainingRunBinding(BundleContractModel):
    """Immutable training/run manifest bound to exact benchmark partitions."""

    schema_version: BundleContractSchemaVersion = BUNDLE_CONTRACT_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    run_version: str = Field(min_length=1)
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    benchmark_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    support_envelope_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    model_artifact: ContentAddressedArtifact
    training_partition_ids: tuple[str, ...] = Field(min_length=1)
    calibration_partition_ids: tuple[str, ...] = ()
    model_selection_validation_partition_ids: tuple[str, ...] = ()
    training_evidence_artifacts: tuple[ContentAddressedArtifact, ...] = Field(min_length=1)
    calibration_evidence_artifacts: tuple[ContentAddressedArtifact, ...] = ()
    model_selection_evidence_artifacts: tuple[ContentAddressedArtifact, ...] = ()
    model_selection_freeze_artifact: ContentAddressedArtifact | None = None

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))

    @field_validator(
        "query_fingerprint",
        "benchmark_fingerprint",
        "support_envelope_fingerprint",
    )
    @classmethod
    def fingerprints_are_lowercase(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def run_is_canonical(self) -> TrainingRunBinding:
        _canonical_text(self.run_id, name="training-run ID")
        _canonical_text(self.run_version, name="training-run version")
        _canonical_values(
            self.training_partition_ids,
            name="training partition IDs",
            allow_empty=False,
        )
        _canonical_values(
            self.calibration_partition_ids,
            name="calibration partition IDs",
        )
        _canonical_values(
            self.model_selection_validation_partition_ids,
            name="model-selection validation partition IDs",
        )
        for artifacts, name, allow_empty in (
            (self.training_evidence_artifacts, "training evidence artifact IDs", False),
            (self.calibration_evidence_artifacts, "calibration evidence artifact IDs", True),
            (
                self.model_selection_evidence_artifacts,
                "model-selection evidence artifact IDs",
                True,
            ),
        ):
            artifact_ids = tuple(item.artifact_id for item in artifacts)
            _canonical_values(artifact_ids, name=name, allow_empty=allow_empty)
        if bool(self.calibration_partition_ids) is not bool(self.calibration_evidence_artifacts):
            raise ValueError(
                "calibration partitions and content-addressed evidence must appear together"
            )
        model_selection_fields = (
            bool(self.model_selection_validation_partition_ids),
            bool(self.model_selection_evidence_artifacts),
            self.model_selection_freeze_artifact is not None,
        )
        if len(set(model_selection_fields)) != 1:
            raise ValueError(
                "model-selection partitions, evidence, and freeze artifact must appear together"
            )
        if model_selection_fields[0] and not self.calibration_partition_ids:
            raise ValueError("model selection cannot precede exact calibration evidence")
        return self


class ValidationEvidenceBinding(BundleContractModel):
    """Typed result manifest tied to exact cases and one implementation scope."""

    schema_version: BundleContractSchemaVersion = BUNDLE_CONTRACT_SCHEMA_VERSION
    evidence_id: str = Field(min_length=1)
    evidence_version: str = Field(min_length=1)
    evidence_kind: ValidationEvidenceKind
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    benchmark_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    support_envelope_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    training_run_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    model_artifact_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    implementation_scope_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    partition_ids: tuple[str, ...] = Field(min_length=1)
    evaluation_case_ids: tuple[str, ...] = Field(min_length=1)
    covered_ports: tuple[BiologicalStagePort, ...] = Field(min_length=1)
    covered_operations: tuple[ModelOperation, ...] = ()
    evidence_artifacts: tuple[ContentAddressedArtifact, ...] = Field(min_length=1)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))

    @property
    def validation_scope_fingerprint(self) -> str:
        """Noncircular identity that a result manifest can bind before its own hash exists."""

        return canonical_fingerprint(self.model_dump(mode="json", exclude={"evidence_artifacts"}))

    @field_validator(
        "query_fingerprint",
        "benchmark_fingerprint",
        "support_envelope_fingerprint",
        "training_run_fingerprint",
        "model_artifact_fingerprint",
        "implementation_scope_fingerprint",
    )
    @classmethod
    def fingerprints_are_lowercase(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def evidence_is_canonical(self) -> ValidationEvidenceBinding:
        _canonical_text(self.evidence_id, name="validation-evidence ID")
        _canonical_text(self.evidence_version, name="validation-evidence version")
        _canonical_values(
            self.partition_ids,
            name="validation partition IDs",
            allow_empty=False,
        )
        _canonical_values(
            self.evaluation_case_ids,
            name="validation evaluation-case IDs",
            allow_empty=False,
        )
        port_values = tuple(port.value for port in self.covered_ports)
        _canonical_values(port_values, name="validation covered ports", allow_empty=False)
        operation_values = tuple(operation.value for operation in self.covered_operations)
        _canonical_values(operation_values, name="validation covered operations")
        artifact_ids = tuple(item.artifact_id for item in self.evidence_artifacts)
        _canonical_values(
            artifact_ids,
            name="validation evidence artifact IDs",
            allow_empty=False,
        )
        return self


class BiologicalModelBundleContract(BundleContractModel):
    """Exhaustive declaration of one component or full biological model bundle."""

    schema_version: BundleContractSchemaVersion = BUNDLE_CONTRACT_SCHEMA_VERSION
    bundle_id: str = Field(min_length=1)
    bundle_version: str = Field(min_length=1)
    bundle_kind: BundleContractKind
    description: str = Field(min_length=1)
    posterior_schema_id: str | None = Field(default=None, min_length=1)
    query: BundleContractReference
    benchmark: BundleContractReference
    support_envelope: BundleContractReference
    model_artifact: ContentAddressedArtifact | None = None
    training_run: BundleContractReference | None = None
    validation_evidence: tuple[BundleContractReference, ...] = ()
    ports: tuple[ModelPortBinding, ...] = Field(min_length=1)
    operation_implementations: tuple[ModelOperationImplementationBinding, ...] = ()

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))

    @property
    def implementation_scope_fingerprint(self) -> str:
        """Hash all implementation-bearing declarations without validation circularity.

        Validation references, validation IDs, rationales, and the bundle fingerprint are
        intentionally excluded.  Changing the trained artifact, any port disposition/code hash/
        interface/entry point, or any high-level operation implementation therefore invalidates
        previously issued validation evidence.
        """

        payload = {
            "bundle_kind": self.bundle_kind.value,
            "posterior_schema_id": self.posterior_schema_id,
            "model_artifact": (
                None if self.model_artifact is None else self.model_artifact.model_dump(mode="json")
            ),
            "ports": [
                {
                    "port": binding.port.value,
                    "disposition": binding.disposition.value,
                    "implementation": (
                        None
                        if binding.implementation is None
                        else binding.implementation.model_dump(mode="json")
                    ),
                }
                for binding in self.ports
            ],
            "operation_implementations": [
                {
                    "operation": binding.operation.value,
                    "implementation": binding.implementation.model_dump(mode="json"),
                }
                for binding in self.operation_implementations
            ],
        }
        return canonical_fingerprint(payload)

    @model_validator(mode="after")
    def map_is_exhaustive_and_canonical(self) -> BiologicalModelBundleContract:
        _canonical_text(self.bundle_id, name="bundle ID")
        _canonical_text(self.bundle_version, name="bundle version")
        _canonical_text(self.description, name="bundle description")
        if self.posterior_schema_id is not None:
            _canonical_text(self.posterior_schema_id, name="posterior schema ID")
        ports = tuple(binding.port for binding in self.ports)
        if len(ports) != len(set(ports)):
            raise ValueError("bundle port map must contain each stage exactly once")
        if set(ports) != set(BiologicalStagePort):
            raise ValueError("bundle port map must classify every biological and support stage")
        if tuple(port.value for port in ports) != tuple(sorted(port.value for port in ports)):
            raise ValueError("bundle port map must be sorted by canonical port name")
        evidence_ids = tuple(item.contract_id for item in self.validation_evidence)
        _canonical_values(evidence_ids, name="bundle validation evidence IDs")
        operations = tuple(item.operation for item in self.operation_implementations)
        if len(operations) != len(set(operations)):
            raise ValueError("bundle runtime operation implementations must be unique")
        if tuple(operation.value for operation in operations) != tuple(
            sorted(operation.value for operation in operations)
        ):
            raise ValueError("bundle runtime operation implementations must be sorted")
        if (
            self.bundle_kind
            in {
                BundleContractKind.COMPONENT_SCAFFOLD,
                BundleContractKind.COMPONENT_MODEL,
            }
            and self.operation_implementations
        ):
            raise ValueError("a direct component cannot bind public runtime implementations")
        if (self.model_artifact is None) is not (self.training_run is None):
            raise ValueError("model artifact and training-run reference must be declared together")
        if self.model_artifact is None and self.validation_evidence:
            raise ValueError("validation evidence cannot exist before a trained model artifact")
        return self


_COMPONENT_PREREQUISITE_REASONS: Mapping[
    BiologicalStagePort,
    QueryPrerequisiteReasonCode,
] = MappingProxyType(
    {
        BiologicalStagePort.ACTION_CONTEXT_ENCODER: (
            QueryPrerequisiteReasonCode.ACTION_CONTEXT_ENCODING
        ),
        BiologicalStagePort.ARTIFACT_PROVENANCE_WRITER: (
            QueryPrerequisiteReasonCode.COMPONENT_PROVENANCE
        ),
        BiologicalStagePort.EXACT_ARTIFACT_RESOLVER: (
            QueryPrerequisiteReasonCode.COMPONENT_ARTIFACT_IDENTITY
        ),
        BiologicalStagePort.POPULATION_ASSAY_RESPONSE_DISTRIBUTION_MODEL: (
            QueryPrerequisiteReasonCode.COMPONENT_RESPONSE_DISTRIBUTION
        ),
        BiologicalStagePort.QUERY_SCOPE_VALIDATOR: (
            QueryPrerequisiteReasonCode.COMPONENT_QUERY_SCOPE
        ),
        BiologicalStagePort.STRICT_SUPPORT_OOD_GATE: (
            QueryPrerequisiteReasonCode.COMPONENT_SUPPORT_OOD
        ),
        BiologicalStagePort.TRAIN_CAL_DATA_LOADER: (
            QueryPrerequisiteReasonCode.COMPONENT_PARTITION_LOADING
        ),
        BiologicalStagePort.UNCERTAINTY_CALIBRATOR: (
            QueryPrerequisiteReasonCode.COMPONENT_UNCERTAINTY_CALIBRATION
        ),
    }
)

QUERY_PREREQUISITE_COMPILER_FINGERPRINT = canonical_fingerprint(
    {
        "compiler_id": QUERY_PREREQUISITE_COMPILER_ID,
        "compiler_version": QUERY_PREREQUISITE_COMPILER_VERSION,
        "algorithm": "exact-query-operation-and-direct-component-port-derivation-v1",
        "operation_floors": {
            operation.value: sorted(port.value for port in ports)
            for operation, ports in OPERATION_REQUIRED_PORTS.items()
        },
        "component_floor": {
            port.value: reason.value for port, reason in _COMPONENT_PREREQUISITE_REASONS.items()
        },
        "conditional_rule_codes": sorted(reason.value for reason in QueryPrerequisiteReasonCode),
        "scope_issue_codes": sorted(issue.value for issue in QueryPrerequisiteScopeIssue),
    }
)


def _add_prerequisite_reason(
    reasons: dict[BiologicalStagePort, set[QueryPrerequisiteReasonCode]],
    port: BiologicalStagePort,
    reason: QueryPrerequisiteReasonCode,
) -> None:
    reasons.setdefault(port, set()).add(reason)


def _canonical_prerequisites(
    reasons: Mapping[BiologicalStagePort, set[QueryPrerequisiteReasonCode]],
) -> tuple[QueryDerivedPortPrerequisite, ...]:
    return tuple(
        QueryDerivedPortPrerequisite(
            port=port,
            reasons=tuple(sorted(port_reasons, key=lambda item: item.value)),
        )
        for port, port_reasons in sorted(reasons.items(), key=lambda item: item[0].value)
    )


def _derive_runtime_target_prerequisites(
    query: StateQuery,
    operation: ModelOperation,
) -> tuple[QueryDerivedPortPrerequisite, ...]:
    reasons: dict[BiologicalStagePort, set[QueryPrerequisiteReasonCode]] = {}
    for port in OPERATION_REQUIRED_PORTS[operation]:
        _add_prerequisite_reason(
            reasons,
            port,
            QueryPrerequisiteReasonCode.OPERATION_CONTRACT_FLOOR,
        )

    # Every public biological result must expose epistemic/model uncertainty, rather than
    # allowing a single fitted model to stand in for calibrated model uncertainty.
    _add_prerequisite_reason(
        reasons,
        BiologicalStagePort.MODEL_ENSEMBLE,
        QueryPrerequisiteReasonCode.EPISTEMIC_MODEL_UNCERTAINTY,
    )

    evidence_roles = set(query.evidence_policy.allowed_evidence_roles)
    evidence_consuming = operation in {
        ModelOperation.ESTIMATE_CELL_STATE,
        ModelOperation.RECOMMEND_NEXT_MEASUREMENT,
    }
    if evidence_consuming:
        _add_prerequisite_reason(
            reasons,
            BiologicalStagePort.OBSERVATION_MODELS,
            QueryPrerequisiteReasonCode.OBSERVATION_LIKELIHOOD,
        )
        if evidence_roles != {EvidenceRole.DIRECT}:
            _add_prerequisite_reason(
                reasons,
                BiologicalStagePort.EVIDENCE_TRANSFER_MODELS,
                QueryPrerequisiteReasonCode.NONDIRECT_EVIDENCE_TRANSFER,
            )

    if any(
        isinstance(output.endpoint, FutureAssayObservationEndpoint)
        for output in query.target_outputs
    ):
        _add_prerequisite_reason(
            reasons,
            BiologicalStagePort.OBSERVATION_MODELS,
            QueryPrerequisiteReasonCode.FUTURE_ASSAY_TARGET,
        )
    if any(output.functional for output in query.target_outputs):
        _add_prerequisite_reason(
            reasons,
            BiologicalStagePort.FUNCTIONAL_DECODERS,
            QueryPrerequisiteReasonCode.FUNCTIONAL_TARGET,
        )

    if query.intervention_space:
        _add_prerequisite_reason(
            reasons,
            BiologicalStagePort.INTERVENTION_REALIZATION_MODEL,
            QueryPrerequisiteReasonCode.INTERVENTION_REALIZATION_UNCERTAINTY,
        )
        if operation is ModelOperation.ESTIMATE_CELL_STATE:
            _add_prerequisite_reason(
                reasons,
                BiologicalStagePort.TRANSITION_MODEL,
                QueryPrerequisiteReasonCode.INTERVENTION_HISTORY_DYNAMICS,
            )

    if query.environment_space and operation is ModelOperation.ESTIMATE_CELL_STATE:
        _add_prerequisite_reason(
            reasons,
            BiologicalStagePort.TRANSITION_MODEL,
            QueryPrerequisiteReasonCode.ENVIRONMENT_HISTORY_DYNAMICS,
        )
    lookback = query.evidence_policy.lookback_seconds
    if operation is ModelOperation.ESTIMATE_CELL_STATE and (lookback is None or lookback > 0):
        _add_prerequisite_reason(
            reasons,
            BiologicalStagePort.TRANSITION_MODEL,
            QueryPrerequisiteReasonCode.TEMPORAL_EVIDENCE_PROPAGATION,
        )

    lineage_roles = {
        EvidenceRole.ANCESTOR,
        EvidenceRole.DESCENDANT,
        EvidenceRole.SIBLING,
        EvidenceRole.CLONE_AGGREGATE,
    }
    lineage_scope = (
        query.subject.kind is SubjectKind.CLONE_LINEAGE
        or query.system_boundary is SystemBoundary.CLONE
        or query.constraints.require_complete_lineage_history
        or bool(evidence_roles & lineage_roles)
    )
    dynamic_operation = operation is not ModelOperation.ESTIMATE_CELL_STATE
    population_dynamics = dynamic_operation and (
        query.subject.kind is SubjectKind.POPULATION
        or query.system_boundary is SystemBoundary.POPULATION
    )
    if lineage_scope or population_dynamics:
        _add_prerequisite_reason(
            reasons,
            BiologicalStagePort.DIVISION_AND_INHERITANCE_MODEL,
            QueryPrerequisiteReasonCode.LINEAGE_OR_POPULATION_DYNAMICS,
        )

    interaction_scope = (
        query.subject.kind is SubjectKind.SPATIAL_NICHE
        or query.system_boundary
        in {
            SystemBoundary.CELL_AND_NEIGHBORS,
            SystemBoundary.SPATIAL_TISSUE_NICHE,
        }
        or query.constraints.require_complete_neighborhood_history
        or EvidenceRole.SPATIAL_NEIGHBOR in evidence_roles
        or population_dynamics
    )
    if interaction_scope:
        _add_prerequisite_reason(
            reasons,
            BiologicalStagePort.CELL_INTERACTION_MODEL,
            QueryPrerequisiteReasonCode.NEIGHBORHOOD_OR_POPULATION_INTERACTIONS,
        )

    transport_scope = query.system_boundary in {
        SystemBoundary.CELL_AND_SOLUBLE_ENVIRONMENT,
        SystemBoundary.SPATIAL_TISSUE_NICHE,
    } or (bool(query.environment_space) and query.constraints.allow_transport)
    if transport_scope:
        _add_prerequisite_reason(
            reasons,
            BiologicalStagePort.EXTRACELLULAR_TRANSPORT_MODEL,
            QueryPrerequisiteReasonCode.SOLUBLE_OR_SPATIAL_TRANSPORT,
        )
    if query.constraints.allow_transport:
        _add_prerequisite_reason(
            reasons,
            BiologicalStagePort.MECHANISTIC_CONSTRAINTS,
            QueryPrerequisiteReasonCode.CAUSAL_TRANSPORT_ASSUMPTIONS,
        )

    declares_mechanism = any(spec.mechanisms for spec in query.intervention_space)
    if declares_mechanism:
        _add_prerequisite_reason(
            reasons,
            BiologicalStagePort.MECHANISTIC_CONSTRAINTS,
            QueryPrerequisiteReasonCode.DECLARED_MECHANISTIC_SEMANTICS,
        )
    if operation in {
        ModelOperation.CHOOSE_INTERVENTION,
        ModelOperation.RECOMMEND_NEXT_MEASUREMENT,
    }:
        _add_prerequisite_reason(
            reasons,
            BiologicalStagePort.SUFFICIENCY_EVALUATOR,
            QueryPrerequisiteReasonCode.PLANNING_STATE_SUFFICIENCY,
        )
        _add_prerequisite_reason(
            reasons,
            BiologicalStagePort.IDENTIFIABILITY_ANALYZER,
            QueryPrerequisiteReasonCode.PLANNING_IDENTIFIABILITY,
        )
        if query.constraints.named_constraints:
            _add_prerequisite_reason(
                reasons,
                BiologicalStagePort.MECHANISTIC_CONSTRAINTS,
                QueryPrerequisiteReasonCode.DECLARED_MECHANISTIC_SEMANTICS,
            )

    if operation is ModelOperation.RECOMMEND_NEXT_MEASUREMENT:
        for port, reason in (
            (
                BiologicalStagePort.OBSERVATION_MODELS,
                QueryPrerequisiteReasonCode.MEASUREMENT_OUTCOME_PREDICTION,
            ),
            (
                BiologicalStagePort.POSTERIOR_INFERENCE_ENGINE,
                QueryPrerequisiteReasonCode.HYPOTHETICAL_MEASUREMENT_UPDATE,
            ),
            (
                BiologicalStagePort.TRANSITION_MODEL,
                QueryPrerequisiteReasonCode.COUNTERFACTUAL_REPLANNING,
            ),
            (
                BiologicalStagePort.FUNCTIONAL_DECODERS,
                QueryPrerequisiteReasonCode.COUNTERFACTUAL_REPLANNING,
            ),
            (
                BiologicalStagePort.VALUE_OF_INFORMATION_ENGINE,
                QueryPrerequisiteReasonCode.MEASUREMENT_DECISION_EVSI,
            ),
        ):
            _add_prerequisite_reason(reasons, port, reason)

    return _canonical_prerequisites(reasons)


def derive_query_prerequisite_report(
    *,
    query: StateQuery,
    support_envelope: BiologicalSupportEnvelope,
    bundle: BiologicalModelBundleContract,
) -> QueryDerivedPrerequisiteReport:
    """Compile exact conditional ports and bind them to query, envelope, and bundle identity."""

    targets: list[QueryDerivedPrerequisiteTarget] = []
    scope_issues: set[QueryPrerequisiteScopeIssue] = set()
    is_component = support_envelope.bundle_kind in {
        BundleContractKind.COMPONENT_SCAFFOLD,
        BundleContractKind.COMPONENT_MODEL,
    }
    if is_component:
        direct_semantics = support_envelope.direct_population_response
        if direct_semantics is not None:
            targets.append(
                QueryDerivedPrerequisiteTarget(
                    target_kind=(QueryPrerequisiteTargetKind.DIRECT_POPULATION_RESPONSE_COMPONENT),
                    direct_population_response=direct_semantics,
                    required_ports=_canonical_prerequisites(
                        {port: {reason} for port, reason in _COMPONENT_PREREQUISITE_REASONS.items()}
                    ),
                )
            )
    else:
        if not support_envelope.runtime_operations:
            scope_issues.add(QueryPrerequisiteScopeIssue.RUNTIME_BUNDLE_REQUIRES_OPERATION)
        measurement_assays = {
            assay.assay_id
            for assay in query.available_assays
            if AssayPurpose.MEASUREMENT_SELECTION in assay.purposes
        }
        for operation in support_envelope.runtime_operations:
            if operation is ModelOperation.CHOOSE_INTERVENTION and not query.intervention_space:
                scope_issues.add(
                    QueryPrerequisiteScopeIssue.INTERVENTION_PLANNING_REQUIRES_ACTION_SPACE
                )
            if operation is ModelOperation.RECOMMEND_NEXT_MEASUREMENT:
                if not query.intervention_space:
                    scope_issues.add(
                        QueryPrerequisiteScopeIssue.MEASUREMENT_EVSI_REQUIRES_ACTION_SPACE
                    )
                if not measurement_assays:
                    scope_issues.add(
                        QueryPrerequisiteScopeIssue.MEASUREMENT_SELECTION_REQUIRES_CANDIDATE_ASSAYS
                    )
            targets.append(
                QueryDerivedPrerequisiteTarget(
                    target_kind=QueryPrerequisiteTargetKind.RUNTIME_OPERATION,
                    operation=operation,
                    required_ports=_derive_runtime_target_prerequisites(query, operation),
                )
            )

    targets_tuple = tuple(sorted(targets, key=lambda item: item.target_id))
    required = {
        prerequisite.port for target in targets_tuple for prerequisite in target.required_ports
    }
    envelope_ports = set(support_envelope.required_ports)
    port_map = {binding.port: binding for binding in bundle.ports}
    invalid_dispositions = tuple(
        QueryPrerequisiteDispositionFailure(
            port=port,
            disposition=port_map[port].disposition,
        )
        for port in sorted(required, key=lambda item: item.value)
        if port_map[port].disposition
        in {
            PortDisposition.PLANNED,
            PortDisposition.NOT_APPLICABLE,
            PortDisposition.UNSUPPORTED,
        }
    )
    return QueryDerivedPrerequisiteReport(
        compiler_id=QUERY_PREREQUISITE_COMPILER_ID,
        compiler_version=QUERY_PREREQUISITE_COMPILER_VERSION,
        compiler_fingerprint=QUERY_PREREQUISITE_COMPILER_FINGERPRINT,
        query_fingerprint=query.fingerprint,
        support_envelope_id=support_envelope.envelope_id,
        support_envelope_version=support_envelope.envelope_version,
        support_envelope_fingerprint=support_envelope.fingerprint,
        bundle_id=bundle.bundle_id,
        bundle_version=bundle.bundle_version,
        bundle_fingerprint=bundle.fingerprint,
        bundle_kind=bundle.bundle_kind,
        targets=targets_tuple,
        required_ports=tuple(sorted(required, key=lambda item: item.value)),
        envelope_missing_ports=tuple(
            sorted(required - envelope_ports, key=lambda item: item.value)
        ),
        envelope_extra_ports=tuple(sorted(envelope_ports - required, key=lambda item: item.value)),
        invalid_dispositions=invalid_dispositions,
        scope_issues=tuple(sorted(scope_issues, key=lambda item: item.value)),
    )


def verify_query_prerequisite_report(
    report: QueryDerivedPrerequisiteReport,
    *,
    query: StateQuery,
    support_envelope: BiologicalSupportEnvelope,
    bundle: BiologicalModelBundleContract,
) -> QueryDerivedPrerequisiteReport:
    """Re-derive a report from sources and reject any caller-supplied drift."""

    derived = derive_query_prerequisite_report(
        query=query,
        support_envelope=support_envelope,
        bundle=bundle,
    )
    if report != derived or report.fingerprint != derived.fingerprint:
        raise ValueError(
            "query-derived prerequisite report does not match exact query/envelope/bundle scope"
        )
    return derived


class BundleReadiness(BundleContractModel):
    """Derived readiness result; no field is accepted on the source bundle contract."""

    bundle_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    lifecycle_stage: ComponentLifecycleStage
    query_binding_verified: bool
    benchmark_binding_verified: bool
    support_envelope_binding_verified: bool
    training_binding_verified: bool
    training_artifacts_verified: bool = False
    training_interfaces_verified: bool = False
    training_result_semantics_verified: bool = False
    trained_candidate_verified: bool = False
    trained_candidate_verification_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[a-fA-F0-9]{64}$",
    )
    calibration_binding_verified: bool
    calibration_result_semantics_verified: bool = False
    model_selection_binding_verified: bool
    model_selection_result_semantics_verified: bool = False
    validation_bindings_verified: bool
    validation_evidence_semantics_verified: bool
    implementation_scope_binding_verified: bool
    artifact_bytes_resolved: bool
    implementation_interfaces_verified: bool
    validation_results_verified: bool
    validation_results_passed: bool
    query_derived_prerequisite_report: QueryDerivedPrerequisiteReport
    query_derived_prerequisite_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    query_derived_prerequisites_verified: bool
    required_ports_provided: bool
    required_ports_executable: bool
    required_ports_evidenced: bool
    benchmark_admission_ready: bool
    component_evaluation_complete: bool
    component_executable: bool
    scientifically_admitted: bool
    component_model_declared: bool
    component_execution_allowed: bool
    runtime_surface_declared: bool
    runtime_operations_bound: bool
    runtime_operations_executable: bool
    runtime_operations_evidenced: bool
    runtime_registration_allowed: bool
    runnable: bool
    admission_blocker_codes: tuple[BundleAdmissionBlockerCode, ...]
    blockers: tuple[str, ...]

    @field_validator(
        "bundle_fingerprint",
        "query_derived_prerequisite_fingerprint",
        "trained_candidate_verification_fingerprint",
    )
    @classmethod
    def fingerprint_is_lowercase(cls, value: str | None) -> str | None:
        return None if value is None else value.casefold()

    @model_validator(mode="after")
    def derived_states_are_consistent(self) -> BundleReadiness:
        if (
            self.query_derived_prerequisite_fingerprint
            != self.query_derived_prerequisite_report.fingerprint
        ):
            raise ValueError(
                "query prerequisite fingerprint must match the embedded derived report"
            )
        if self.bundle_fingerprint != self.query_derived_prerequisite_report.bundle_fingerprint:
            raise ValueError("query prerequisite report must bind this exact bundle")
        if self.query_derived_prerequisites_verified and not (
            self.query_derived_prerequisite_report.structurally_satisfied
        ):
            raise ValueError("query prerequisites cannot verify with scope or disposition failures")
        if self.validation_results_passed and not self.validation_results_verified:
            raise ValueError("validation results cannot pass before exact semantic verification")
        expected_trained_candidate = all(
            (
                self.query_binding_verified,
                self.benchmark_binding_verified,
                self.support_envelope_binding_verified,
                self.training_binding_verified,
                self.training_artifacts_verified,
                self.training_interfaces_verified,
                self.training_result_semantics_verified,
                self.query_derived_prerequisites_verified,
            )
        )
        if self.trained_candidate_verified is not expected_trained_candidate:
            raise ValueError(
                "trained candidate must derive from exact p1 bytes, interface, and semantics"
            )
        if (self.trained_candidate_verification_fingerprint is not None) is not (
            self.trained_candidate_verified
        ):
            raise ValueError(
                "trained-candidate verification fingerprint must appear only with verification"
            )
        if self.calibration_result_semantics_verified and not (
            self.trained_candidate_verified and self.calibration_binding_verified
        ):
            raise ValueError("calibration semantics cannot verify before a trained candidate")
        if self.model_selection_result_semantics_verified and not (
            self.calibration_result_semantics_verified and self.model_selection_binding_verified
        ):
            raise ValueError("model-selection semantics cannot verify before calibration")
        blocker_code_values = tuple(code.value for code in self.admission_blocker_codes)
        _canonical_values(blocker_code_values, name="admission blocker codes")
        expected_codes = {
            code
            for flag, code in (
                (
                    self.artifact_bytes_resolved,
                    BundleAdmissionBlockerCode.ARTIFACT_BYTES_UNRESOLVED,
                ),
                (
                    self.implementation_interfaces_verified,
                    BundleAdmissionBlockerCode.IMPLEMENTATION_INTERFACES_UNVERIFIED,
                ),
                (
                    self.validation_results_verified,
                    BundleAdmissionBlockerCode.VALIDATION_RESULTS_UNVERIFIED,
                ),
                (
                    self.query_derived_prerequisites_verified,
                    BundleAdmissionBlockerCode.QUERY_DERIVED_OPERATION_PREREQUISITES_UNVERIFIED,
                ),
            )
            if not flag
        }
        if self.validation_results_verified and not self.validation_results_passed:
            expected_codes.add(BundleAdmissionBlockerCode.VALIDATION_RESULTS_FAILED)
        if set(self.admission_blocker_codes) != expected_codes:
            raise ValueError("admission blocker codes must exactly match derived trust gates")
        expected_component = all(
            (
                self.query_binding_verified,
                self.benchmark_binding_verified,
                self.support_envelope_binding_verified,
                self.training_binding_verified,
                self.trained_candidate_verified,
                self.calibration_binding_verified,
                self.calibration_result_semantics_verified,
                self.model_selection_binding_verified,
                self.model_selection_result_semantics_verified,
                self.validation_bindings_verified,
                self.validation_evidence_semantics_verified,
                self.implementation_scope_binding_verified,
                self.artifact_bytes_resolved,
                self.implementation_interfaces_verified,
                self.validation_results_passed,
                self.query_derived_prerequisites_verified,
                self.required_ports_provided,
                self.required_ports_executable,
                self.required_ports_evidenced,
                self.component_evaluation_complete,
            )
        )
        if self.component_executable is not expected_component:
            raise ValueError(
                "component executability must be derived from exact bindings and ports"
            )
        expected_admission = self.component_executable and self.benchmark_admission_ready
        if self.scientifically_admitted is not expected_admission:
            raise ValueError("scientific admission must include executable component and benchmark")
        expected_component_execution = (
            self.scientifically_admitted and self.component_model_declared
        )
        if self.component_execution_allowed is not expected_component_execution:
            raise ValueError("component execution requires an admitted component-model surface")
        expected_registration = self.scientifically_admitted and all(
            (
                self.runtime_surface_declared,
                self.runtime_operations_bound,
                self.runtime_operations_executable,
                self.runtime_operations_evidenced,
                self.query_derived_prerequisites_verified,
            )
        )
        if self.runtime_registration_allowed is not expected_registration:
            raise ValueError("runtime registration must include an exact executable API surface")
        if self.runnable is not self.runtime_registration_allowed:
            raise ValueError(
                "a biological bundle is runnable only when runtime registration passes"
            )
        execution_allowed = self.runnable or self.component_execution_allowed
        if execution_allowed is bool(self.blockers):
            raise ValueError("derived execution state and blockers are inconsistent")
        expected_lifecycle = ComponentLifecycleStage.SCAFFOLD
        if self.trained_candidate_verified:
            expected_lifecycle = ComponentLifecycleStage.TRAINED_CANDIDATE
        if (
            expected_lifecycle is ComponentLifecycleStage.TRAINED_CANDIDATE
            and self.calibration_binding_verified
            and self.calibration_result_semantics_verified
        ):
            expected_lifecycle = ComponentLifecycleStage.CALIBRATED_CANDIDATE
        if (
            expected_lifecycle is ComponentLifecycleStage.CALIBRATED_CANDIDATE
            and self.model_selection_binding_verified
            and self.model_selection_result_semantics_verified
        ):
            expected_lifecycle = ComponentLifecycleStage.MODEL_SELECTED_FROZEN
        if (
            expected_lifecycle is ComponentLifecycleStage.MODEL_SELECTED_FROZEN
            and self.component_evaluation_complete
        ):
            expected_lifecycle = ComponentLifecycleStage.COMPONENT_EVALUATED
        if (
            expected_lifecycle is ComponentLifecycleStage.COMPONENT_EVALUATED
            and self.scientifically_admitted
        ):
            expected_lifecycle = ComponentLifecycleStage.COMPONENT_GATES_PASSED
        if self.lifecycle_stage is not expected_lifecycle:
            raise ValueError("component lifecycle must be derived from exact verified evidence")
        if (
            self.runnable
            and self.lifecycle_stage is not ComponentLifecycleStage.COMPONENT_GATES_PASSED
        ):
            raise ValueError("a runnable bundle must have passed component gates")
        return self


def _reference_matches(
    reference: BundleContractReference,
    *,
    contract_id: str,
    contract_version: str,
    payload: SchemaModel,
) -> bool:
    payload_bytes = canonical_json_bytes(payload.model_dump(mode="json"))
    return (
        reference.contract_id == contract_id
        and reference.contract_version == contract_version
        and reference.artifact.sha256 == canonical_fingerprint(payload.model_dump(mode="json"))
        and reference.artifact.byte_count == len(payload_bytes)
        and reference.artifact.media_type == "application/json"
    )


def _append_once(blockers: list[str], blocker: str) -> None:
    if blocker not in blockers:
        blockers.append(blocker)


def _benchmark_evaluation_complete(benchmark: BenchmarkArtifact) -> bool:
    """Return whether every frozen executable evaluation was run, whether or not it passed."""

    definition = benchmark.definition
    admission = benchmark.admission
    audit = benchmark.leakage_audit
    if (
        audit is None
        or any(check.status is AuditCheckStatus.NOT_ASSESSED for check in audit.checks)
        or not definition.metrics
        or not definition.baselines
    ):
        return False
    if any(
        not isinstance(metric.implementation_binding, ExecutableImplementationBinding)
        or not isinstance(metric.uncertainty.method, ExecutableImplementationBinding)
        for metric in definition.metrics
    ) or any(
        not isinstance(baseline.implementation_binding, ExecutableImplementationBinding)
        for baseline in definition.baselines
        if baseline.applicability.applies_to(definition.query.state_query)
    ):
        return False
    expected_results = {
        (metric.metric_id, partition_id)
        for metric in definition.metrics
        for partition_id in metric.evaluation_partition_ids
    }
    actual_results = {
        (result.metric.metric_id, result.partition_id) for result in admission.metric_results
    }
    if expected_results != actual_results:
        return False
    applicable_runs = tuple(
        run
        for baseline, run in zip(definition.baselines, admission.baseline_runs, strict=True)
        if baseline.applicability.applies_to(definition.query.state_query)
    )
    if not applicable_runs or any(
        run.status is BaselineRunStatus.NOT_RUN for run in applicable_runs
    ):
        return False
    if any(rule.baseline_comparator is not None for rule in definition.acceptance_rules):
        return bool(admission.paired_baseline_comparisons)
    return True


def assess_biological_model_bundle(
    bundle: BiologicalModelBundleContract,
    *,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    manifests: Mapping[str, DatasetManifest],
    support_envelope: BiologicalSupportEnvelope,
    training_run: TrainingRunBinding | None = None,
    validation_evidence: Sequence[ValidationEvidenceBinding] = (),
    admission_context: AdmissionVerificationContext | None = None,
    training_context: TrainingVerificationContext | None = None,
) -> BundleReadiness:
    """Derive readiness from source declarations plus an optional runtime-only trust context.

    Without an external context every trust-bearing state remains false.  With one, this function
    re-authenticates exact bytes, interfaces, semantic results, and query prerequisites; serialized
    readiness is never accepted as execution authority.
    """

    blockers: list[str] = []
    artifact_bytes_resolved = False
    implementation_interfaces_verified = False
    validation_results_verified = False
    validation_results_passed = False
    training_artifacts_verified = False
    training_interfaces_verified = False
    training_result_semantics_verified = False
    trained_candidate_verified = False
    trained_candidate_verification_fingerprint: str | None = None
    calibration_result_semantics_verified = False
    model_selection_result_semantics_verified = False
    prerequisite_report = derive_query_prerequisite_report(
        query=query,
        support_envelope=support_envelope,
        bundle=bundle,
    )
    query_derived_prerequisites_verified = False
    if admission_context is None:
        _append_once(blockers, "trusted artifact-byte receipts are absent")
        _append_once(blockers, "trusted loaded-interface receipts are absent")
        _append_once(blockers, "trusted typed validation-result semantics are absent")
        if training_context is None:
            _append_once(blockers, "trusted query-prerequisite binding is absent")
    else:
        from .validation import require_exact_query_prerequisites

        try:
            require_exact_query_prerequisites(
                admission_context,
                query=query,
                support_envelope=support_envelope,
                bundle=bundle,
            )
        except ValueError as error:
            _append_once(blockers, f"query-prerequisite verification failed: {error}")
        else:
            query_derived_prerequisites_verified = True
    for port in prerequisite_report.envelope_missing_ports:
        _append_once(
            blockers,
            f"query-derived prerequisite port {port.value} is absent from the support envelope",
        )
    for port in prerequisite_report.envelope_extra_ports:
        _append_once(
            blockers,
            f"support-envelope required port {port.value} is not required by the exact query "
            "target",
        )
    for failure in prerequisite_report.invalid_dispositions:
        _append_once(
            blockers,
            f"query-derived prerequisite port {failure.port.value} is {failure.disposition.value}",
        )
    for issue in prerequisite_report.scope_issues:
        _append_once(blockers, f"query prerequisite scope issue: {issue.value}")
    benchmark_query = benchmark.definition.query
    query_binding_verified = (
        query.fingerprint == benchmark_query.query_fingerprint
        and query == benchmark_query.state_query
        and _reference_matches(
            bundle.query,
            contract_id=benchmark_query.query_id,
            contract_version=benchmark_query.query_version,
            payload=query,
        )
        and bundle.query == support_envelope.query
    )
    if not query_binding_verified:
        _append_once(blockers, "exact query binding did not verify")

    benchmark_binding_verified = (
        _reference_matches(
            bundle.benchmark,
            contract_id=benchmark.definition.benchmark_id,
            contract_version=benchmark.definition.benchmark_version,
            payload=benchmark,
        )
        and bundle.benchmark == support_envelope.benchmark
    )
    if not benchmark_binding_verified:
        _append_once(blockers, "exact benchmark binding did not verify")

    support_envelope_binding_verified = (
        bundle.bundle_kind is support_envelope.bundle_kind
        and _reference_matches(
            bundle.support_envelope,
            contract_id=support_envelope.envelope_id,
            contract_version=support_envelope.envelope_version,
            payload=support_envelope,
        )
    )
    if not support_envelope_binding_verified:
        _append_once(blockers, "exact support-envelope binding did not verify")

    try:
        benchmark_verification = verify_benchmark_artifact(benchmark, manifests)
    except (KeyError, ValueError) as error:
        benchmark_admission_ready = False
        _append_once(blockers, f"benchmark verification failed: {error}")
    else:
        benchmark_admission_ready = benchmark_verification.admission_ready
        if not benchmark_admission_ready:
            _append_once(blockers, "bound benchmark is not scientifically admitted")

    if admission_context is not None:
        from .validation import require_exact_admission_receipts

        try:
            receipt_report = require_exact_admission_receipts(
                admission_context,
                bundle=bundle,
                benchmark=benchmark,
                training_run=training_run,
                validation_evidence=validation_evidence,
            )
        except ValueError as error:
            _append_once(blockers, f"admission receipt verification failed: {error}")
        else:
            artifact_bytes_resolved = True
            implementation_interfaces_verified = receipt_report.required_interfaces == () or len(
                receipt_report.interface_receipts
            ) == len(receipt_report.required_interfaces)

    port_map = {binding.port: binding for binding in bundle.ports}
    required = set(support_envelope.required_ports)
    extra_required = {
        port
        for port, binding in port_map.items()
        if binding.disposition is PortDisposition.REQUIRED and port not in required
    }
    required_ports_provided = not extra_required and all(
        port_map[port].disposition is PortDisposition.PROVIDED for port in required
    )
    for port in sorted(extra_required, key=lambda item: item.value):
        _append_once(
            blockers,
            f"port {port.value} is marked required outside the exact support envelope",
        )
    if not required_ports_provided:
        for port in sorted(required, key=lambda item: item.value):
            binding = port_map[port]
            if binding.disposition is not PortDisposition.PROVIDED:
                _append_once(
                    blockers,
                    f"required port {port.value} is {binding.disposition.value}",
                )

    implementations = tuple(port_map[port].implementation for port in required)
    required_ports_executable = all(
        (
            required_ports_provided,
            artifact_bytes_resolved,
            implementation_interfaces_verified,
            all(
                implementation is not None and implementation.declares_python_entry_point
                for implementation in implementations
            ),
        )
    )
    if required_ports_provided and not required_ports_executable:
        for port in sorted(required, key=lambda item: item.value):
            implementation = port_map[port].implementation
            if implementation is None or not implementation.declares_python_entry_point:
                _append_once(
                    blockers,
                    f"required port {port.value} has no Python entry-point declaration",
                )

    training_binding_verified = False
    calibration_binding_verified = False
    model_selection_binding_verified = False
    if bundle.model_artifact is None or bundle.training_run is None or training_run is None:
        _append_once(blockers, "trained model artifact and exact training-run binding are absent")
    else:
        if training_context is None:
            plan = benchmark.definition.split_plan
            training_ids = tuple(
                sorted(
                    partition.partition_id
                    for partition in (() if plan is None else plan.partitions)
                    if partition.role is BenchmarkPartitionRole.TRAIN
                )
            )
            calibration_ids = tuple(
                sorted(
                    partition.partition_id
                    for partition in (() if plan is None else plan.partitions)
                    if partition.role is BenchmarkPartitionRole.CALIBRATION
                )
            )
            model_selection_ids = tuple(
                sorted(
                    partition.partition_id
                    for partition in (() if plan is None else plan.partitions)
                    if partition.role is BenchmarkPartitionRole.MODEL_SELECTION_VALIDATION
                )
            )
        else:
            training_ids = training_context.plan.training_partition_ids
            calibration_ids = ()
            model_selection_ids = ()
        training_binding_verified = (
            _reference_matches(
                bundle.training_run,
                contract_id=training_run.run_id,
                contract_version=training_run.run_version,
                payload=training_run,
            )
            and training_run.query_fingerprint == query.fingerprint
            and training_run.benchmark_fingerprint == benchmark.fingerprint
            and training_run.support_envelope_fingerprint == support_envelope.fingerprint
            and training_run.model_artifact == bundle.model_artifact
            and training_run.training_partition_ids == training_ids
            and training_run.calibration_partition_ids in ((), calibration_ids)
            and training_run.model_selection_validation_partition_ids in ((), model_selection_ids)
        )
        if not training_binding_verified:
            _append_once(blockers, "exact training/model/partition binding did not verify")
        else:
            calibration_binding_verified = (
                training_run.calibration_partition_ids == calibration_ids
                and bool(training_run.calibration_evidence_artifacts)
            )
            model_selection_binding_verified = (
                calibration_binding_verified
                and training_run.model_selection_validation_partition_ids == model_selection_ids
                and bool(training_run.model_selection_evidence_artifacts)
                and training_run.model_selection_freeze_artifact is not None
            )
    if training_binding_verified and training_context is not None and training_run is not None:
        from .training import require_exact_trained_candidate

        try:
            trained_verification = require_exact_trained_candidate(
                bundle,
                query=query,
                benchmark=benchmark,
                support_envelope=support_envelope,
                training_run=training_run,
                context=training_context,
            )
        except ValueError as error:
            _append_once(blockers, f"trained-candidate verification failed: {error}")
        else:
            training_artifacts_verified = True
            training_interfaces_verified = True
            training_result_semantics_verified = True
            trained_candidate_verified = True
            trained_candidate_verification_fingerprint = (
                trained_verification.verification_fingerprint
            )
            query_derived_prerequisites_verified = True
    if training_binding_verified and not trained_candidate_verified:
        _append_once(blockers, "trusted trained-candidate verification is absent")
    if training_binding_verified and not calibration_binding_verified:
        _append_once(blockers, "exact calibration evidence is absent")
    if calibration_binding_verified and not model_selection_binding_verified:
        _append_once(blockers, "exact model-selection freeze evidence is absent")

    declared_validation = {item.contract_id: item for item in bundle.validation_evidence}
    actual_validation = {item.evidence_id: item for item in validation_evidence}
    evidence_requirements = {
        item.evidence_id: item for item in support_envelope.required_validation_evidence
    }
    required_evidence = set(evidence_requirements)
    validation_bindings_verified = bool(required_evidence) and (
        set(declared_validation) == set(actual_validation) == required_evidence
    )
    if (
        validation_bindings_verified
        and training_run is not None
        and bundle.model_artifact is not None
    ):
        for evidence_id in sorted(required_evidence):
            evidence = actual_validation[evidence_id]
            reference = declared_validation[evidence_id]
            if not (
                _reference_matches(
                    reference,
                    contract_id=evidence.evidence_id,
                    contract_version=evidence.evidence_version,
                    payload=evidence,
                )
                and evidence.query_fingerprint == query.fingerprint
                and evidence.benchmark_fingerprint == benchmark.fingerprint
                and evidence.support_envelope_fingerprint == support_envelope.fingerprint
                and evidence.training_run_fingerprint == training_run.fingerprint
                and evidence.model_artifact_fingerprint == bundle.model_artifact.sha256
            ):
                validation_bindings_verified = False
                break
    else:
        validation_bindings_verified = False
    if not validation_bindings_verified:
        _append_once(blockers, "required exact validation-evidence bindings are absent or invalid")

    implementation_scope_binding_verified = validation_bindings_verified and all(
        evidence.implementation_scope_fingerprint == bundle.implementation_scope_fingerprint
        for evidence in actual_validation.values()
    )
    if not implementation_scope_binding_verified:
        _append_once(
            blockers,
            "validation evidence does not bind the exact model/port/operation implementation scope",
        )

    plan = benchmark.definition.split_plan
    partitions = () if plan is None else plan.partitions
    case_set = benchmark.definition.evaluation_case_set
    validation_evidence_semantics_verified = validation_bindings_verified
    if validation_evidence_semantics_verified:
        for evidence_id in sorted(required_evidence):
            evidence = actual_validation[evidence_id]
            requirement = evidence_requirements[evidence_id]
            expected_partition_ids = tuple(
                sorted(
                    partition.partition_id
                    for partition in partitions
                    if partition.role in requirement.partition_roles
                )
            )
            expected_case_ids = tuple(
                sorted(
                    case.case_id
                    for case in (() if case_set is None else case_set.cases)
                    if case.partition_id in expected_partition_ids
                )
            )
            if (
                evidence.evidence_kind is not requirement.evidence_kind
                or not expected_partition_ids
                or evidence.partition_ids != expected_partition_ids
                or not expected_case_ids
                or evidence.evaluation_case_ids != expected_case_ids
            ):
                validation_evidence_semantics_verified = False
                break
    if not validation_evidence_semantics_verified:
        _append_once(
            blockers,
            "validation evidence kind, partition role, or evaluation-case identity is invalid",
        )

    if (
        admission_context is not None
        and artifact_bytes_resolved
        and validation_evidence_semantics_verified
        and implementation_scope_binding_verified
    ):
        from .validation import require_exact_admission_validation_results

        try:
            result_batch = require_exact_admission_validation_results(
                admission_context,
                bundle=bundle,
                support_envelope=support_envelope,
                validation_evidence=validation_evidence,
            )
        except ValueError as error:
            _append_once(blockers, f"validation-result verification failed: {error}")
        else:
            validation_results_verified = True
            validation_results_passed = result_batch.all_results_passed
            if not validation_results_passed:
                _append_once(blockers, "one or more authenticated validation results did not pass")

    evidence_coverage = {
        evidence_id: set(evidence.covered_ports)
        for evidence_id, evidence in actual_validation.items()
    }
    required_ports_evidenced = all(
        (
            validation_bindings_verified,
            validation_evidence_semantics_verified,
            implementation_scope_binding_verified,
            validation_results_passed,
        )
    ) and all(
        port_map[port].validation_evidence_ids
        and set(port_map[port].validation_evidence_ids) <= required_evidence
        and all(
            port in evidence_coverage[evidence_id]
            for evidence_id in port_map[port].validation_evidence_ids
        )
        for port in required
    )
    if not required_ports_evidenced:
        _append_once(blockers, "required ports lack exact validation evidence coverage")

    operation_map = {binding.operation: binding for binding in bundle.operation_implementations}
    declared_operations = set(support_envelope.runtime_operations)
    runtime_operations_bound = set(operation_map) == declared_operations
    if not runtime_operations_bound:
        _append_once(
            blockers,
            "runtime operation implementations do not exactly match the support envelope",
        )
    runtime_operations_executable = all(
        (
            bool(declared_operations),
            runtime_operations_bound,
            artifact_bytes_resolved,
            implementation_interfaces_verified,
            all(
                binding.implementation.declares_python_entry_point
                for binding in operation_map.values()
            ),
        )
    )
    if declared_operations and runtime_operations_bound and not runtime_operations_executable:
        _append_once(blockers, "one or more public runtime operations are specification-only")

    operation_evidence_coverage = {
        evidence_id: set(evidence.covered_operations)
        for evidence_id, evidence in actual_validation.items()
    }
    runtime_operations_evidenced = all(
        (
            validation_bindings_verified,
            validation_evidence_semantics_verified,
            implementation_scope_binding_verified,
            validation_results_passed,
        )
    ) and all(
        operation_map[operation].validation_evidence_ids
        and set(operation_map[operation].validation_evidence_ids) <= required_evidence
        and all(
            operation in operation_evidence_coverage[evidence_id]
            for evidence_id in operation_map[operation].validation_evidence_ids
        )
        for operation in declared_operations
    )
    if declared_operations and not runtime_operations_evidenced:
        _append_once(blockers, "public runtime operations lack exact validation evidence coverage")

    component_evaluation_complete = (
        model_selection_binding_verified
        and model_selection_result_semantics_verified
        and validation_bindings_verified
        and validation_evidence_semantics_verified
        and implementation_scope_binding_verified
        and artifact_bytes_resolved
        and implementation_interfaces_verified
        and validation_results_verified
        and _benchmark_evaluation_complete(benchmark)
    )
    if model_selection_binding_verified and not component_evaluation_complete:
        _append_once(blockers, "complete executable component evaluation is absent")

    component_executable = all(
        (
            query_binding_verified,
            benchmark_binding_verified,
            support_envelope_binding_verified,
            training_binding_verified,
            trained_candidate_verified,
            calibration_binding_verified,
            calibration_result_semantics_verified,
            model_selection_binding_verified,
            model_selection_result_semantics_verified,
            validation_bindings_verified,
            validation_evidence_semantics_verified,
            implementation_scope_binding_verified,
            artifact_bytes_resolved,
            implementation_interfaces_verified,
            validation_results_passed,
            query_derived_prerequisites_verified,
            required_ports_provided,
            required_ports_executable,
            required_ports_evidenced,
            component_evaluation_complete,
        )
    )
    scientifically_admitted = component_executable and benchmark_admission_ready
    component_model_declared = bundle.bundle_kind is BundleContractKind.COMPONENT_MODEL
    component_execution_allowed = scientifically_admitted and component_model_declared
    lifecycle_stage = ComponentLifecycleStage.SCAFFOLD
    if trained_candidate_verified:
        lifecycle_stage = ComponentLifecycleStage.TRAINED_CANDIDATE
    if (
        lifecycle_stage is ComponentLifecycleStage.TRAINED_CANDIDATE
        and calibration_binding_verified
        and calibration_result_semantics_verified
    ):
        lifecycle_stage = ComponentLifecycleStage.CALIBRATED_CANDIDATE
    if (
        lifecycle_stage is ComponentLifecycleStage.CALIBRATED_CANDIDATE
        and model_selection_binding_verified
        and model_selection_result_semantics_verified
    ):
        lifecycle_stage = ComponentLifecycleStage.MODEL_SELECTED_FROZEN
    if (
        lifecycle_stage is ComponentLifecycleStage.MODEL_SELECTED_FROZEN
        and component_evaluation_complete
    ):
        lifecycle_stage = ComponentLifecycleStage.COMPONENT_EVALUATED
    if lifecycle_stage is ComponentLifecycleStage.COMPONENT_EVALUATED and scientifically_admitted:
        lifecycle_stage = ComponentLifecycleStage.COMPONENT_GATES_PASSED
    runtime_surface_declared = (
        bundle.bundle_kind is BundleContractKind.BIOLOGICAL_MODEL_BUNDLE
        and bool(support_envelope.runtime_operations)
        and bundle.posterior_schema_id is not None
    )
    runtime_registration_allowed = scientifically_admitted and all(
        (
            runtime_surface_declared,
            runtime_operations_bound,
            runtime_operations_executable,
            runtime_operations_evidenced,
            query_derived_prerequisites_verified,
        )
    )
    if bundle.bundle_kind is BundleContractKind.COMPONENT_SCAFFOLD:
        _append_once(blockers, "component scaffold is not a public cell-state runtime backend")
    elif bundle.bundle_kind is BundleContractKind.COMPONENT_MODEL:
        if not component_execution_allowed:
            _append_once(blockers, "direct component has not passed its execution gates")
    elif not support_envelope.runtime_operations:
        _append_once(blockers, "support envelope registers no public runtime operations")
    elif bundle.posterior_schema_id is None:
        _append_once(blockers, "biological runtime bundle lacks a posterior schema binding")

    admission_blocker_code_set: set[BundleAdmissionBlockerCode] = set()
    if not artifact_bytes_resolved:
        admission_blocker_code_set.add(BundleAdmissionBlockerCode.ARTIFACT_BYTES_UNRESOLVED)
    if not implementation_interfaces_verified:
        admission_blocker_code_set.add(
            BundleAdmissionBlockerCode.IMPLEMENTATION_INTERFACES_UNVERIFIED
        )
    if not validation_results_verified:
        admission_blocker_code_set.add(BundleAdmissionBlockerCode.VALIDATION_RESULTS_UNVERIFIED)
    elif not validation_results_passed:
        admission_blocker_code_set.add(BundleAdmissionBlockerCode.VALIDATION_RESULTS_FAILED)
    if not query_derived_prerequisites_verified:
        admission_blocker_code_set.add(
            BundleAdmissionBlockerCode.QUERY_DERIVED_OPERATION_PREREQUISITES_UNVERIFIED
        )
    admission_blocker_codes = tuple(sorted(admission_blocker_code_set, key=lambda code: code.value))
    blockers = sorted(set(blockers))
    return BundleReadiness(
        bundle_fingerprint=bundle.fingerprint,
        lifecycle_stage=lifecycle_stage,
        query_binding_verified=query_binding_verified,
        benchmark_binding_verified=benchmark_binding_verified,
        support_envelope_binding_verified=support_envelope_binding_verified,
        training_binding_verified=training_binding_verified,
        training_artifacts_verified=training_artifacts_verified,
        training_interfaces_verified=training_interfaces_verified,
        training_result_semantics_verified=training_result_semantics_verified,
        trained_candidate_verified=trained_candidate_verified,
        trained_candidate_verification_fingerprint=(trained_candidate_verification_fingerprint),
        calibration_binding_verified=calibration_binding_verified,
        calibration_result_semantics_verified=calibration_result_semantics_verified,
        model_selection_binding_verified=model_selection_binding_verified,
        model_selection_result_semantics_verified=model_selection_result_semantics_verified,
        validation_bindings_verified=validation_bindings_verified,
        validation_evidence_semantics_verified=validation_evidence_semantics_verified,
        implementation_scope_binding_verified=implementation_scope_binding_verified,
        artifact_bytes_resolved=artifact_bytes_resolved,
        implementation_interfaces_verified=implementation_interfaces_verified,
        validation_results_verified=validation_results_verified,
        validation_results_passed=validation_results_passed,
        query_derived_prerequisite_report=prerequisite_report,
        query_derived_prerequisite_fingerprint=prerequisite_report.fingerprint,
        query_derived_prerequisites_verified=query_derived_prerequisites_verified,
        required_ports_provided=required_ports_provided,
        required_ports_executable=required_ports_executable,
        required_ports_evidenced=required_ports_evidenced,
        benchmark_admission_ready=benchmark_admission_ready,
        component_evaluation_complete=component_evaluation_complete,
        component_executable=component_executable,
        scientifically_admitted=scientifically_admitted,
        component_model_declared=component_model_declared,
        component_execution_allowed=component_execution_allowed,
        runtime_surface_declared=runtime_surface_declared,
        runtime_operations_bound=runtime_operations_bound,
        runtime_operations_executable=runtime_operations_executable,
        runtime_operations_evidenced=runtime_operations_evidenced,
        runtime_registration_allowed=runtime_registration_allowed,
        runnable=runtime_registration_allowed,
        admission_blocker_codes=admission_blocker_codes,
        blockers=tuple(blockers),
    )


class BiologicalExecutionBlockedError(RuntimeError):
    """Raised when code attempts biological execution before derived admission."""


def _require_derived_readiness(readiness: BundleReadiness) -> None:
    if not readiness.runnable:
        reasons = "; ".join(readiness.blockers) or "biological execution is not admitted"
        raise BiologicalExecutionBlockedError(reasons)


def require_biological_execution(
    bundle: BiologicalModelBundleContract,
    *,
    operation: ModelOperation,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    manifests: Mapping[str, DatasetManifest],
    support_envelope: BiologicalSupportEnvelope,
    training_run: TrainingRunBinding | None = None,
    validation_evidence: Sequence[ValidationEvidenceBinding] = (),
    admission_context: AdmissionVerificationContext | None = None,
    jit_code_provider: JITCodeProvider | None = None,
) -> BiologicalExecutionAuthorization:
    """Re-derive admission and fail closed before every biological runtime execution.

    Accepting source artifacts here rather than a caller-created readiness object prevents a
    hand-constructed Boolean report from becoming an execution authorization.
    """

    readiness = assess_biological_model_bundle(
        bundle,
        query=query,
        benchmark=benchmark,
        manifests=manifests,
        support_envelope=support_envelope,
        training_run=training_run,
        validation_evidence=validation_evidence,
        admission_context=admission_context,
    )
    _require_derived_readiness(readiness)
    bound_operations = {binding.operation for binding in bundle.operation_implementations}
    if operation not in support_envelope.runtime_operations or operation not in bound_operations:
        raise BiologicalExecutionBlockedError(
            f"{operation.value} is not an exact admitted operation of this bundle"
        )
    if admission_context is None or jit_code_provider is None:
        raise BiologicalExecutionBlockedError(
            "biological execution requires an authenticated context and JIT code provider"
        )
    from .validation import BiologicalExecutionAuthorization, reverify_admission_jit_interfaces

    try:
        handles = reverify_admission_jit_interfaces(
            admission_context,
            provider=jit_code_provider,
            operation=operation,
        )
    except ValueError as error:
        raise BiologicalExecutionBlockedError(
            f"JIT interface verification failed: {error}"
        ) from error
    return BiologicalExecutionAuthorization(
        readiness=readiness,
        operation=operation,
        runtime_handles=handles,
    )


def require_biological_component_execution(
    bundle: BiologicalModelBundleContract,
    *,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    manifests: Mapping[str, DatasetManifest],
    support_envelope: BiologicalSupportEnvelope,
    training_run: TrainingRunBinding | None = None,
    validation_evidence: Sequence[ValidationEvidenceBinding] = (),
    admission_context: AdmissionVerificationContext | None = None,
    jit_code_provider: JITCodeProvider | None = None,
) -> BiologicalExecutionAuthorization:
    """Authorize only the direct component surface, never a public belief operation."""

    readiness = assess_biological_model_bundle(
        bundle,
        query=query,
        benchmark=benchmark,
        manifests=manifests,
        support_envelope=support_envelope,
        training_run=training_run,
        validation_evidence=validation_evidence,
        admission_context=admission_context,
    )
    if not readiness.component_execution_allowed:
        reasons = "; ".join(readiness.blockers) or "component execution is not admitted"
        raise BiologicalExecutionBlockedError(reasons)
    if admission_context is None or jit_code_provider is None:
        raise BiologicalExecutionBlockedError(
            "component execution requires an authenticated context and JIT code provider"
        )
    from .validation import BiologicalExecutionAuthorization, reverify_admission_jit_interfaces

    try:
        handles = reverify_admission_jit_interfaces(
            admission_context,
            provider=jit_code_provider,
            operation=None,
        )
    except ValueError as error:
        raise BiologicalExecutionBlockedError(
            f"JIT interface verification failed: {error}"
        ) from error
    return BiologicalExecutionAuthorization(
        readiness=readiness,
        operation=None,
        runtime_handles=handles,
    )


def build_admitted_estimator_descriptor(
    bundle: BiologicalModelBundleContract,
    *,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
    manifests: Mapping[str, DatasetManifest],
    support_envelope: BiologicalSupportEnvelope,
    training_run: TrainingRunBinding | None = None,
    validation_evidence: Sequence[ValidationEvidenceBinding] = (),
    admission_context: AdmissionVerificationContext | None = None,
    jit_code_provider: JITCodeProvider | None = None,
) -> EstimatorDescriptor:
    """Bridge an admitted full bundle to the existing public runtime protocol descriptor."""

    require_biological_execution(
        bundle,
        operation=ModelOperation.ESTIMATE_CELL_STATE,
        query=query,
        benchmark=benchmark,
        manifests=manifests,
        support_envelope=support_envelope,
        training_run=training_run,
        validation_evidence=validation_evidence,
        admission_context=admission_context,
        jit_code_provider=jit_code_provider,
    )
    if (
        bundle.bundle_kind is not BundleContractKind.BIOLOGICAL_MODEL_BUNDLE
        or bundle.model_artifact is None
        or bundle.training_run is None
        or bundle.posterior_schema_id is None
    ):
        raise BiologicalExecutionBlockedError("only an admitted full bundle has a descriptor")
    return EstimatorDescriptor(
        model_id=bundle.bundle_id,
        model_version=bundle.bundle_version,
        model_fingerprint=bundle.model_artifact.sha256,
        posterior_schema_id=bundle.posterior_schema_id,
        description=bundle.description,
        artifact_kind=ModelArtifactKind.BIOLOGICAL_MODEL,
        support_envelope_id=bundle.support_envelope.contract_id,
        support_envelope_fingerprint=bundle.support_envelope.artifact.sha256,
        training_support_id=bundle.training_run.contract_id,
        training_support_fingerprint=bundle.training_run.artifact.sha256,
        validation_evidence_ids=tuple(item.contract_id for item in bundle.validation_evidence),
        validation_evidence_fingerprints={
            item.contract_id: item.artifact.sha256 for item in bundle.validation_evidence
        },
    )


__all__ = [
    "BUNDLE_CONTRACT_SCHEMA_VERSION",
    "OPERATION_REQUIRED_PORTS",
    "ORIGINAL_SKELETON_PORTS",
    "RUNTIME_FOUNDATION_PORTS",
    "BiologicalExecutionBlockedError",
    "BiologicalModelBundleContract",
    "BiologicalStagePort",
    "BiologicalSupportEnvelope",
    "BundleAdmissionBlockerCode",
    "BundleContractKind",
    "BundleContractReference",
    "BundleReadiness",
    "ComponentLifecycleStage",
    "DirectPopulationResponseSemantics",
    "ModelOperation",
    "ModelOperationImplementationBinding",
    "ModelPortBinding",
    "PortDisposition",
    "PortImplementationBinding",
    "PortImplementationKind",
    "TrainingRunBinding",
    "ValidationEvidenceBinding",
    "ValidationEvidenceKind",
    "ValidationEvidenceRequirement",
    "assess_biological_model_bundle",
    "build_admitted_estimator_descriptor",
    "require_biological_component_execution",
    "require_biological_execution",
]
