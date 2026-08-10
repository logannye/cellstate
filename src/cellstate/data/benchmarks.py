"""Fail-closed contracts for immutable, query-scoped biological benchmarks.

A benchmark definition is not an admission decision. Definitions freeze task semantics and bytes;
admission additionally requires re-resolving scientific and legal evidence, leakage checks, and
baseline execution. Random record splits cannot stand in for independent experimental units.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import date
from enum import StrEnum
from typing import Annotated, Literal, TypeAlias
from urllib.parse import urlparse

from pydantic import ConfigDict, Field, field_validator, model_validator

from cellstate.domain.common import (
    CausalStatus,
    OntologyTerm,
    SchemaModel,
    canonical_fingerprint,
    canonical_json_bytes,
    require_finite,
)
from cellstate.domain.query import StateQuery, SystemBoundary
from cellstate.domain.subjects import SubjectKind, TargetAggregation
from cellstate.training.objectives import LossKind

from .manifests import (
    AssessmentKind,
    ClaimAssessment,
    DatasetAssessmentReference,
    DatasetAssessmentResolution,
    DatasetManifest,
    DataUseCase,
    EligibilityStatus,
    ExperimentalUnitLevel,
    IdentificationBasis,
    LossEligibilityAssessment,
    MetricEligibilityAssessment,
    MetricFamily,
    MetricPartitionPurpose,
    ObjectiveEligibilityAssessment,
    PermissionStatus,
    ScientificClaim,
    UnitIdentityExpression,
)

BenchmarkSchemaVersion = Literal["0.1-experimental"]
BENCHMARK_SCHEMA_VERSION: BenchmarkSchemaVersion = "0.1-experimental"
QueryBindingSchemaVersion = Literal["0.1-experimental"]
QUERY_BINDING_SCHEMA_VERSION: QueryBindingSchemaVersion = "0.1-experimental"
SplitPlanSchemaVersion = Literal["0.1-experimental"]
SPLIT_PLAN_SCHEMA_VERSION: SplitPlanSchemaVersion = "0.1-experimental"
EvaluationCaseSetSchemaVersion = Literal["0.1-experimental"]
EVALUATION_CASE_SET_SCHEMA_VERSION: EvaluationCaseSetSchemaVersion = "0.1-experimental"
MetricDefinitionSchemaVersion = Literal["0.1-experimental"]
METRIC_DEFINITION_SCHEMA_VERSION: MetricDefinitionSchemaVersion = "0.1-experimental"
BaselineDefinitionSchemaVersion = Literal["0.1-experimental"]
BASELINE_DEFINITION_SCHEMA_VERSION: BaselineDefinitionSchemaVersion = "0.1-experimental"
LeakageAuditSchemaVersion = Literal["0.1-experimental"]
LEAKAGE_AUDIT_SCHEMA_VERSION: LeakageAuditSchemaVersion = "0.1-experimental"
AdmissionSchemaVersion = Literal["0.1-experimental"]
ADMISSION_SCHEMA_VERSION: AdmissionSchemaVersion = "0.1-experimental"

ScalarParameterValue: TypeAlias = str | int | float | bool


class BenchmarkModel(SchemaModel):
    """Strict model used at every benchmark artifact boundary."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        revalidate_instances="always",
        allow_inf_nan=False,
        strict=True,
    )


def _canonical_id(value: str, *, name: str) -> str:
    if not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a canonical nonempty string")
    return value


def _unique_sorted(values: tuple[str, ...], *, name: str, allow_empty: bool = True) -> None:
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    for value in values:
        _canonical_id(value, name=name)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")
    if tuple(sorted(values)) != values:
        raise ValueError(f"{name} must be sorted")


def _fingerprint(model: SchemaModel) -> str:
    return canonical_fingerprint(model.model_dump(mode="json"))


def _canonical_model_bytes(model: SchemaModel) -> bytes:
    if isinstance(model, DatasetManifest):
        return model.canonical_json_bytes
    return canonical_json_bytes(model.model_dump(mode="json"))


class ContentAddressedArtifact(BenchmarkModel):
    artifact_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    byte_count: int = Field(gt=0)
    media_type: str = Field(min_length=1)

    @field_validator("artifact_id")
    @classmethod
    def artifact_id_is_canonical(cls, value: str) -> str:
        return _canonical_id(value, name="artifact ID")

    @field_validator("sha256")
    @classmethod
    def canonicalize_sha256(cls, value: str) -> str:
        return value.casefold()

    @field_validator("uri")
    @classmethod
    def uri_is_immutable_and_remote(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme.casefold() not in {"https", "http", "s3", "gs", "drs"}:
            raise ValueError("benchmark artifacts require an absolute remote URI")
        if not parsed.netloc or parsed.username is not None or parsed.password is not None:
            raise ValueError("benchmark artifact URI must be public, absolute, and credential-free")
        if (parsed.hostname or "").casefold() in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("benchmark artifact URI must not resolve to localhost")
        return value


class VersionedImplementation(BenchmarkModel):
    implementation_id: str = Field(min_length=1)
    implementation_version: str = Field(min_length=1)
    code_artifact: ContentAddressedArtifact
    entrypoint: str = Field(min_length=1)
    runtime: str = Field(min_length=1)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @model_validator(mode="after")
    def identity_is_canonical(self) -> VersionedImplementation:
        _canonical_id(self.implementation_id, name="implementation ID")
        _canonical_id(self.implementation_version, name="implementation version")
        _canonical_id(self.entrypoint, name="implementation entrypoint")
        _canonical_id(self.runtime, name="implementation runtime")
        return self


class BenchmarkParameter(BenchmarkModel):
    name: str = Field(min_length=1)
    value: ScalarParameterValue

    @field_validator("name")
    @classmethod
    def name_is_canonical(cls, value: str) -> str:
        return _canonical_id(value, name="parameter name")

    @field_validator("value")
    @classmethod
    def value_is_finite(cls, value: ScalarParameterValue) -> ScalarParameterValue:
        if isinstance(value, float):
            require_finite(value, name="benchmark parameter")
        return value


def _parameters_are_canonical(parameters: tuple[BenchmarkParameter, ...]) -> None:
    names = tuple(parameter.name for parameter in parameters)
    _unique_sorted(names, name="parameter names")


class QueryParameterValue(BenchmarkModel):
    value_id: str = Field(min_length=1)
    canonical_json: str = Field(min_length=1)

    @model_validator(mode="after")
    def value_is_canonical_json(self) -> QueryParameterValue:
        _canonical_id(self.value_id, name="query-grid value ID")
        try:
            parsed = json.loads(self.canonical_json)
        except json.JSONDecodeError as error:
            raise ValueError("query-grid value must be valid JSON") from error
        if isinstance(parsed, float) and not math.isfinite(parsed):
            raise ValueError("query-grid numeric values must be finite")
        encoded = json.dumps(
            parsed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if encoded != self.canonical_json:
            raise ValueError("query-grid value must use canonical compact JSON")
        return self


class QueryParameterAxis(BenchmarkModel):
    axis_id: str = Field(min_length=1)
    query_path: str = Field(min_length=1)
    values: tuple[QueryParameterValue, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def axis_is_canonical(self) -> QueryParameterAxis:
        _canonical_id(self.axis_id, name="query-grid axis ID")
        _canonical_id(self.query_path, name="query-grid path")
        value_ids = tuple(value.value_id for value in self.values)
        _unique_sorted(value_ids, name="query-grid value IDs", allow_empty=False)
        return self


class QueryParameterGrid(BenchmarkModel):
    grid_id: str = Field(min_length=1)
    grid_version: str = Field(min_length=1)
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    authoritative: Literal[False] = False
    axes: tuple[QueryParameterAxis, ...] = Field(min_length=1)
    allowed_combinations_artifact: ContentAddressedArtifact | None = None

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @field_validator("query_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def grid_is_canonical(self) -> QueryParameterGrid:
        _canonical_id(self.grid_id, name="query-grid ID")
        _canonical_id(self.grid_version, name="query-grid version")
        axis_ids = tuple(axis.axis_id for axis in self.axes)
        _unique_sorted(axis_ids, name="query-grid axis IDs", allow_empty=False)
        return self


class StateQueryBinding(BenchmarkModel):
    schema_version: QueryBindingSchemaVersion = QUERY_BINDING_SCHEMA_VERSION
    query_id: str = Field(min_length=1)
    query_version: str = Field(min_length=1)
    query_schema_version: Literal["2.0"]
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    query_artifact: ContentAddressedArtifact
    state_query: StateQuery
    parameter_grid: QueryParameterGrid | None = None

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @field_validator("query_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def exact_payload_is_bound(self) -> StateQueryBinding:
        _canonical_id(self.query_id, name="query ID")
        _canonical_id(self.query_version, name="query version")
        if self.query_schema_version != self.state_query.schema_version:
            raise ValueError("declared query schema version must match the embedded StateQuery")
        if self.query_fingerprint != self.state_query.fingerprint:
            raise ValueError("query fingerprint must match the exact embedded StateQuery")
        canonical_bytes = _canonical_model_bytes(self.state_query)
        if self.query_artifact.sha256 != self.query_fingerprint:
            raise ValueError("query artifact must contain the canonical StateQuery bytes")
        if self.query_artifact.byte_count != len(canonical_bytes):
            raise ValueError("query artifact byte count must match canonical StateQuery bytes")
        if self.query_artifact.media_type != "application/json":
            raise ValueError("canonical StateQuery artifact must use application/json")
        if (
            self.parameter_grid is not None
            and self.parameter_grid.query_fingerprint != self.query_fingerprint
        ):
            raise ValueError("query parameter grid must bind the exact StateQuery")
        return self


class BenchmarkScope(BenchmarkModel):
    scope_id: str = Field(min_length=1)
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    subject_kind: SubjectKind
    system_boundary: SystemBoundary
    biological_system: OntologyTerm
    target_output_keys: tuple[str, ...] = Field(min_length=1)
    horizon_names: tuple[str, ...] = Field(min_length=1)
    intervention_spec_ids: tuple[str, ...] = ()
    scientific_claims: tuple[ScientificClaim, ...] = ()
    inference_cutoff_seconds: float | None = None
    inference_cutoff_field: str | None = Field(default=None, min_length=1)
    reference_estimand_causal_status: CausalStatus
    forecast_causal_status: CausalStatus
    estimand: str = Field(min_length=1)

    @field_validator("query_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @field_validator("inference_cutoff_seconds")
    @classmethod
    def cutoff_is_finite(cls, value: float | None) -> float | None:
        if value is not None:
            return require_finite(value, name="benchmark inference cutoff")
        return value

    @model_validator(mode="after")
    def scope_is_canonical(self) -> BenchmarkScope:
        _canonical_id(self.scope_id, name="benchmark scope ID")
        _canonical_id(self.estimand, name="benchmark estimand")
        for values, name in (
            (self.target_output_keys, "scope target-output keys"),
            (self.horizon_names, "scope horizon names"),
            (self.intervention_spec_ids, "scope intervention IDs"),
        ):
            _unique_sorted(values, name=name, allow_empty=name == "scope intervention IDs")
        claims = tuple(claim.value for claim in self.scientific_claims)
        _unique_sorted(claims, name="scope scientific claims")
        if (self.inference_cutoff_seconds is None) is (self.inference_cutoff_field is None):
            raise ValueError("benchmark scope requires exactly one fixed or field cutoff")
        if self.inference_cutoff_field is not None:
            _canonical_id(self.inference_cutoff_field, name="benchmark inference-cutoff field")
        causal_claims = set(self.scientific_claims)
        for status in (
            self.reference_estimand_causal_status,
            self.forecast_causal_status,
        ):
            if (
                status is CausalStatus.IDENTIFIED_POPULATION_EFFECT
                and ScientificClaim.INTERVENTION_EFFECT not in causal_claims
            ):
                raise ValueError("identified-effect status requires an intervention-effect claim")
            if (
                status is CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS
                and ScientificClaim.COUNTERFACTUAL_GENERALIZATION not in causal_claims
            ):
                raise ValueError(
                    "transported status requires a counterfactual-generalization claim"
                )
        return self


class ClaimAssessmentIdentity(BenchmarkModel):
    assessment_kind: Literal[AssessmentKind.CLAIM] = AssessmentKind.CLAIM
    claim: ScientificClaim


class LossAssessmentIdentity(BenchmarkModel):
    assessment_kind: Literal[AssessmentKind.LOSS] = AssessmentKind.LOSS
    loss_kind: LossKind


class MetricAssessmentIdentity(BenchmarkModel):
    assessment_kind: Literal[AssessmentKind.METRIC] = AssessmentKind.METRIC
    metric_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.-]*$")
    metric_family: MetricFamily
    partition_purpose: MetricPartitionPurpose

    @field_validator("metric_id")
    @classmethod
    def metric_id_is_canonical(cls, value: str) -> str:
        return _canonical_id(value, name="metric-assessment identity ID")


AssessmentIdentity: TypeAlias = Annotated[
    ClaimAssessmentIdentity | LossAssessmentIdentity | MetricAssessmentIdentity,
    Field(discriminator="assessment_kind"),
]


class BenchmarkEvidenceBinding(BenchmarkModel):
    binding_id: str = Field(min_length=1)
    physical_dataset_binding_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    manifest_artifact: ContentAddressedArtifact
    manifest_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    assessment_reference: DatasetAssessmentReference
    assessment_kind: AssessmentKind
    assessment_identity: AssessmentIdentity
    scope_binding: EvidenceScopeBinding
    required_split_unit: ExperimentalUnitBinding
    representability_proof_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[a-fA-F0-9]{64}$",
    )

    @field_validator("manifest_fingerprint", "representability_proof_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str | None) -> str | None:
        return value.casefold() if value is not None else None

    @model_validator(mode="after")
    def reference_is_exact(self) -> BenchmarkEvidenceBinding:
        _canonical_id(self.binding_id, name="benchmark evidence-binding ID")
        _canonical_id(
            self.physical_dataset_binding_id,
            name="physical dataset-binding ID",
        )
        _canonical_id(self.dataset_id, name="benchmark dataset ID")
        _canonical_id(self.dataset_version, name="benchmark dataset version")
        if self.assessment_identity.assessment_kind is not self.assessment_kind:
            raise ValueError("typed assessment identity must match the assessment kind")
        if isinstance(self.assessment_identity, ClaimAssessmentIdentity) and (
            self.scope_binding.scientific_claims != (self.assessment_identity.claim,)
        ):
            raise ValueError("claim identity must match the exact evidence claim class")
        if self.assessment_reference.dataset_manifest_fingerprint != self.manifest_fingerprint:
            raise ValueError("assessment reference must bind the exact dataset manifest")
        if (
            self.manifest_artifact.sha256 != self.manifest_fingerprint
            or self.manifest_artifact.media_type != "application/json"
        ):
            raise ValueError("manifest artifact must contain canonical manifest JSON bytes")
        return self


class ExperimentalUnitBinding(BenchmarkModel):
    level: ExperimentalUnitLevel
    identity: UnitIdentityExpression

    @property
    def key(self) -> str:
        return f"{self.level.value}:{self.identity.fingerprint}"


class EvidenceTargetMapping(BenchmarkModel):
    """Exact query target mapped to the assessment members that realize it."""

    target_output_key: str = Field(min_length=1)
    target_output_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    target_units: str = Field(min_length=1)
    target_aggregation: TargetAggregation
    aggregation_unit: ExperimentalUnitBinding
    assessment_modalities: tuple[str, ...] = ()
    assessment_functional_readout_ids: tuple[str, ...] = ()
    semantics_artifact: ContentAddressedArtifact

    @field_validator("target_output_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def mapping_is_exact(self) -> EvidenceTargetMapping:
        _canonical_id(self.target_output_key, name="target-evidence output key")
        _canonical_id(self.target_units, name="target-evidence units")
        _unique_sorted(self.assessment_modalities, name="target-evidence modalities")
        _unique_sorted(
            self.assessment_functional_readout_ids,
            name="target-evidence functional readouts",
        )
        if not self.assessment_modalities and not self.assessment_functional_readout_ids:
            raise ValueError("each target mapping requires a modality or functional readout")
        return self


class EvidenceInterventionMapping(BenchmarkModel):
    intervention_spec_id: str = Field(min_length=1)
    intervention_spec_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    assessment_intervention_kind_key: str = Field(min_length=1)
    domain_mapping_artifact: ContentAddressedArtifact

    @field_validator("intervention_spec_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def mapping_is_canonical(self) -> EvidenceInterventionMapping:
        _canonical_id(self.intervention_spec_id, name="intervention-evidence spec ID")
        _canonical_id(
            self.assessment_intervention_kind_key,
            name="assessment intervention-kind key",
        )
        return self


class EvidenceEnvironmentMapping(BenchmarkModel):
    environment_variable_key: str = Field(min_length=1)
    environment_spec_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    assessment_environment_variable_key: str = Field(min_length=1)
    domain_mapping_artifact: ContentAddressedArtifact

    @field_validator("environment_spec_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def mapping_is_canonical(self) -> EvidenceEnvironmentMapping:
        _canonical_id(self.environment_variable_key, name="environment-evidence variable key")
        _canonical_id(
            self.assessment_environment_variable_key,
            name="assessment environment-variable key",
        )
        return self


class EvidenceScopeBinding(BenchmarkModel):
    """Typed exact map from one frozen query to one exact assessment scope."""

    assessment_scope_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    target_mappings: tuple[EvidenceTargetMapping, ...] = Field(min_length=1)
    intervention_mappings: tuple[EvidenceInterventionMapping, ...] = ()
    environment_mappings: tuple[EvidenceEnvironmentMapping, ...] = ()
    horizon_names: tuple[str, ...] = Field(min_length=1)
    scientific_claims: tuple[ScientificClaim, ...] = ()

    @field_validator("assessment_scope_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def members_are_canonical(self) -> EvidenceScopeBinding:
        target_keys = tuple(item.target_output_key for item in self.target_mappings)
        intervention_ids = tuple(item.intervention_spec_id for item in self.intervention_mappings)
        environment_keys = tuple(
            item.environment_variable_key for item in self.environment_mappings
        )
        _unique_sorted(target_keys, name="evidence target mappings", allow_empty=False)
        _unique_sorted(intervention_ids, name="evidence intervention mappings")
        _unique_sorted(environment_keys, name="evidence environment mappings")
        _unique_sorted(self.horizon_names, name="evidence horizon names", allow_empty=False)
        _unique_sorted(
            tuple(claim.value for claim in self.scientific_claims),
            name="evidence scientific claims",
        )
        return self


BenchmarkEvidenceBinding.model_rebuild()


class ProtectedGroupReason(StrEnum):
    DEFAULT_SPLIT = "default_split"
    SAMPLING_SUBJECT = "sampling_subject"
    BIOLOGICAL_REPLICATE = "biological_replicate"
    RANDOMIZATION = "randomization"
    OBJECTIVE_REQUIRED_SPLIT = "objective_required_split"
    METRIC_EVALUATION = "metric_evaluation"
    SPLIT_ASSIGNMENT = "split_assignment"


class ProtectedGroupBinding(BenchmarkModel):
    unit: ExperimentalUnitBinding
    reasons: tuple[ProtectedGroupReason, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reasons_are_canonical(self) -> ProtectedGroupBinding:
        values = tuple(reason.value for reason in self.reasons)
        _unique_sorted(values, name="protected-group reasons", allow_empty=False)
        return self


class ProtectedGroupClosure(BenchmarkModel):
    physical_dataset_binding_id: str = Field(min_length=1)
    unit_ancestry: tuple[ExperimentalUnitBinding, ...] = Field(min_length=1)
    record_unit: ExperimentalUnitBinding
    assignment_unit: ExperimentalUnitBinding
    protected_groups: tuple[ProtectedGroupBinding, ...] = Field(min_length=1)
    forbid_random_record_splitting: Literal[True] = True
    forbid_shared_protected_groups: Literal[True] = True

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @model_validator(mode="after")
    def closure_is_safe_and_explicit(self) -> ProtectedGroupClosure:
        _canonical_id(
            self.physical_dataset_binding_id,
            name="closure physical dataset-binding ID",
        )
        ancestry_keys = tuple(unit.key for unit in self.unit_ancestry)
        _unique_sorted(
            tuple(sorted(ancestry_keys)),
            name="experimental-unit ancestry members",
            allow_empty=False,
        )
        if len(ancestry_keys) != len(set(ancestry_keys)):
            raise ValueError("experimental-unit ancestry must not repeat units")
        if self.record_unit != self.unit_ancestry[-1]:
            raise ValueError("record unit must be the finest declared ancestry member")
        if self.assignment_unit not in self.unit_ancestry:
            raise ValueError("split assignment unit must occur in the unit ancestry")
        groups = {group.unit.key: group for group in self.protected_groups}
        if len(groups) != len(self.protected_groups):
            raise ValueError("protected-group units must be unique")
        if tuple(sorted(groups)) != tuple(group.unit.key for group in self.protected_groups):
            raise ValueError("protected groups must be sorted by unit key")
        if any(group.unit not in self.unit_ancestry for group in self.protected_groups):
            raise ValueError("every protected group must occur in the unit ancestry")
        assignment_group = groups.get(self.assignment_unit.key)
        if assignment_group is None or (
            ProtectedGroupReason.SPLIT_ASSIGNMENT not in assignment_group.reasons
        ):
            raise ValueError("assignment unit must be an explicit protected group")
        assignment_index = self.unit_ancestry.index(self.assignment_unit)
        if any(
            self.unit_ancestry.index(group.unit) < assignment_index
            for group in self.protected_groups
        ):
            raise ValueError("split assignment cannot be finer than a protected group")
        return self


class CanonicalIdMembership(BenchmarkModel):
    ids_artifact: ContentAddressedArtifact
    encoding: Literal["canonical_json_utf8_string_array_v1"] = "canonical_json_utf8_string_array_v1"
    id_count: int = Field(gt=0)


class EvaluationCaseRole(StrEnum):
    TREATED = "treated"
    MATCHED_NO_ACTION_CONTROL = "matched_no_action_control"


class EvaluationContextBinding(BenchmarkModel):
    context_id: str = Field(min_length=1)
    context_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    context_artifact: ContentAddressedArtifact

    @field_validator("context_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def context_is_content_addressed(self) -> EvaluationContextBinding:
        _canonical_id(self.context_id, name="evaluation context ID")
        if (
            self.context_artifact.sha256 != self.context_fingerprint
            or self.context_artifact.media_type != "application/json"
        ):
            raise ValueError("evaluation context artifact must match its exact fingerprint")
        return self


class BenchmarkEvaluationCase(BenchmarkModel):
    case_id: str = Field(min_length=1)
    partition_id: str = Field(min_length=1)
    evaluation_unit_id: str = Field(min_length=1)
    prediction_subject_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    context_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    matching_stratum_id: str = Field(min_length=1)
    matching_stratum_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    intervention_spec_ids: tuple[str, ...] = Field(max_length=1)
    horizon_name: str = Field(min_length=1)
    target_output_keys: tuple[str, ...] = Field(min_length=1)
    role: EvaluationCaseRole
    matched_control_evaluation_unit_ids: tuple[str, ...] = ()

    @field_validator("context_fingerprint", "matching_stratum_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def case_is_exact(self) -> BenchmarkEvaluationCase:
        for value, name in (
            (self.case_id, "evaluation-case ID"),
            (self.partition_id, "evaluation-case partition"),
            (self.evaluation_unit_id, "evaluation-unit ID"),
            (self.prediction_subject_id, "prediction-subject ID"),
            (self.context_id, "evaluation context ID"),
            (self.matching_stratum_id, "matching-stratum ID"),
            (self.horizon_name, "evaluation horizon"),
        ):
            _canonical_id(value, name=name)
        _unique_sorted(self.intervention_spec_ids, name="case intervention spec IDs")
        _unique_sorted(
            self.target_output_keys,
            name="case target-output keys",
            allow_empty=False,
        )
        _unique_sorted(
            self.matched_control_evaluation_unit_ids,
            name="case matched-control unit IDs",
        )
        if self.role is EvaluationCaseRole.TREATED:
            if len(self.intervention_spec_ids) != 1:
                raise ValueError("treated cases require exactly one intervention spec")
            if not self.matched_control_evaluation_unit_ids:
                raise ValueError("treated cases require exact matched-control unit IDs")
            if self.evaluation_unit_id in self.matched_control_evaluation_unit_ids:
                raise ValueError("a treated unit cannot serve as its own matched control")
        elif self.intervention_spec_ids or self.matched_control_evaluation_unit_ids:
            raise ValueError("matched no-action controls require zero actions and no control refs")
        return self


class EvaluationCasePartitionBinding(BenchmarkModel):
    partition_id: str = Field(min_length=1)
    evaluation_unit_ids: CanonicalIdMembership

    @field_validator("partition_id")
    @classmethod
    def partition_id_is_canonical(cls, value: str) -> str:
        return _canonical_id(value, name="case-set partition ID")


class EvaluationInterventionMultiplicity(BenchmarkModel):
    intervention_spec_id: str = Field(min_length=1)
    case_count: int = Field(gt=0)

    @field_validator("intervention_spec_id")
    @classmethod
    def intervention_id_is_canonical(cls, value: str) -> str:
        return _canonical_id(value, name="case-set intervention spec ID")


class BenchmarkEvaluationCaseSet(BenchmarkModel):
    schema_version: EvaluationCaseSetSchemaVersion = EVALUATION_CASE_SET_SCHEMA_VERSION
    case_set_id: str = Field(min_length=1)
    case_set_version: str = Field(min_length=1)
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    evaluation_unit: ExperimentalUnitBinding
    contexts: tuple[EvaluationContextBinding, ...] = Field(min_length=1)
    case_artifact: ContentAddressedArtifact
    cases_encoding: Literal["canonical_json_utf8_evaluation_case_array_v1"] = (
        "canonical_json_utf8_evaluation_case_array_v1"
    )
    case_count: int = Field(gt=0)
    partition_memberships: tuple[EvaluationCasePartitionBinding, ...] = Field(min_length=1)
    intervention_case_counts: tuple[EvaluationInterventionMultiplicity, ...] = ()
    no_action_control_case_count: int = Field(ge=0)
    cases: tuple[BenchmarkEvaluationCase, ...] = Field(min_length=1)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @field_validator("query_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def set_is_complete_and_content_addressed(self) -> BenchmarkEvaluationCaseSet:
        _canonical_id(self.case_set_id, name="evaluation-case-set ID")
        _canonical_id(self.case_set_version, name="evaluation-case-set version")
        context_ids = tuple(item.context_id for item in self.contexts)
        case_ids = tuple(item.case_id for item in self.cases)
        unit_ids = tuple(item.evaluation_unit_id for item in self.cases)
        partition_ids = tuple(item.partition_id for item in self.partition_memberships)
        intervention_ids = tuple(
            item.intervention_spec_id for item in self.intervention_case_counts
        )
        _unique_sorted(context_ids, name="evaluation context IDs", allow_empty=False)
        _unique_sorted(case_ids, name="evaluation-case IDs", allow_empty=False)
        for unit_id in unit_ids:
            _canonical_id(unit_id, name="evaluation-case unit ID")
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("evaluation-case unit IDs must be unique")
        _unique_sorted(partition_ids, name="case-set partition IDs", allow_empty=False)
        _unique_sorted(intervention_ids, name="case-set intervention IDs")
        if self.case_count != len(self.cases):
            raise ValueError("evaluation case count must match the embedded exact cases")

        case_bytes = canonical_json_bytes([case.model_dump(mode="json") for case in self.cases])
        if (
            self.case_artifact.sha256
            != canonical_fingerprint([case.model_dump(mode="json") for case in self.cases])
            or self.case_artifact.byte_count != len(case_bytes)
            or self.case_artifact.media_type != "application/json"
        ):
            raise ValueError("case artifact must contain the exact canonical case array bytes")

        contexts = {item.context_id: item for item in self.contexts}
        cases_by_unit = {item.evaluation_unit_id: item for item in self.cases}
        cases_by_partition: dict[str, list[BenchmarkEvaluationCase]] = {
            partition_id: [] for partition_id in partition_ids
        }
        actual_intervention_counts: dict[str, int] = {}
        actual_control_count = 0
        for case in self.cases:
            context = contexts.get(case.context_id)
            if context is None or context.context_fingerprint != case.context_fingerprint:
                raise ValueError("every evaluation case must bind one exact declared context")
            if case.partition_id not in cases_by_partition:
                raise ValueError("evaluation case references an undeclared partition")
            cases_by_partition[case.partition_id].append(case)
            if case.role is EvaluationCaseRole.MATCHED_NO_ACTION_CONTROL:
                actual_control_count += 1
            else:
                spec_id = case.intervention_spec_ids[0]
                actual_intervention_counts[spec_id] = actual_intervention_counts.get(spec_id, 0) + 1

        declared_intervention_counts = {
            item.intervention_spec_id: item.case_count for item in self.intervention_case_counts
        }
        if declared_intervention_counts != actual_intervention_counts:
            raise ValueError("declared intervention multiplicities must exactly match cases")
        if self.no_action_control_case_count != actual_control_count:
            raise ValueError("declared no-action control count must exactly match cases")

        for binding in self.partition_memberships:
            ids = tuple(
                sorted(case.evaluation_unit_id for case in cases_by_partition[binding.partition_id])
            )
            ids_bytes = canonical_json_bytes(ids)
            if (
                binding.evaluation_unit_ids.id_count != len(ids)
                or binding.evaluation_unit_ids.ids_artifact.sha256 != canonical_fingerprint(ids)
                or binding.evaluation_unit_ids.ids_artifact.byte_count != len(ids_bytes)
                or binding.evaluation_unit_ids.ids_artifact.media_type != "application/json"
            ):
                raise ValueError(
                    "case-set partition membership must contain exact evaluation-unit IDs"
                )

        for case in self.cases:
            for control_id in case.matched_control_evaluation_unit_ids:
                control = cases_by_unit.get(control_id)
                if control is None:
                    raise ValueError("treated case references an unknown matched control")
                if control.role is not EvaluationCaseRole.MATCHED_NO_ACTION_CONTROL:
                    raise ValueError("treated case controls must be explicit no-action cases")
                if (
                    control.partition_id != case.partition_id
                    or control.context_id != case.context_id
                    or control.context_fingerprint != case.context_fingerprint
                    or control.matching_stratum_id != case.matching_stratum_id
                    or control.matching_stratum_fingerprint != case.matching_stratum_fingerprint
                ):
                    raise ValueError("matched controls must share exact context and stratum")
        return self


class ProtectedGroupMembership(BenchmarkModel):
    unit: ExperimentalUnitBinding
    membership: CanonicalIdMembership


class ExplicitPartitionMembership(BenchmarkModel):
    schema_version: SplitPlanSchemaVersion = SPLIT_PLAN_SCHEMA_VERSION
    kind: Literal["explicit_membership"] = "explicit_membership"
    assignment_unit: ExperimentalUnitBinding
    assignment_unit_ids: CanonicalIdMembership
    record_unit: ExperimentalUnitBinding
    record_ids: CanonicalIdMembership
    descendant_closure_artifact: ContentAddressedArtifact
    protected_group_memberships: tuple[ProtectedGroupMembership, ...] = Field(min_length=1)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @model_validator(mode="after")
    def memberships_are_canonical(self) -> ExplicitPartitionMembership:
        keys = tuple(item.unit.key for item in self.protected_group_memberships)
        _unique_sorted(keys, name="protected-group membership units", allow_empty=False)
        if self.assignment_unit_ids.id_count > self.record_ids.id_count:
            raise ValueError("assignment-unit count cannot exceed descendant record count")
        return self


class PartitionGenerationSpec(BenchmarkModel):
    schema_version: SplitPlanSchemaVersion = SPLIT_PLAN_SCHEMA_VERSION
    kind: Literal["generated_membership"] = "generated_membership"
    generator: VersionedImplementation
    source_universe_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    assignment_unit: ExperimentalUnitBinding
    seed: int = Field(ge=0)
    parameters: tuple[BenchmarkParameter, ...] = ()
    stratification_fields: tuple[str, ...] = ()
    materialized_membership: ExplicitPartitionMembership | None = None

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @field_validator("source_universe_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def generation_is_canonical(self) -> PartitionGenerationSpec:
        _parameters_are_canonical(self.parameters)
        _unique_sorted(self.stratification_fields, name="split stratification fields")
        if (
            self.materialized_membership is not None
            and self.materialized_membership.assignment_unit != self.assignment_unit
        ):
            raise ValueError("generated output must use the declared assignment unit")
        return self


PartitionMembershipSpec: TypeAlias = Annotated[
    ExplicitPartitionMembership | PartitionGenerationSpec,
    Field(discriminator="kind"),
]


class ExcludedPartitionMembership(BenchmarkModel):
    assignment_unit_ids: CanonicalIdMembership
    record_ids: CanonicalIdMembership
    descendant_closure_artifact: ContentAddressedArtifact
    reason_codes_artifact: ContentAddressedArtifact

    @model_validator(mode="after")
    def exclusion_counts_are_coherent(self) -> ExcludedPartitionMembership:
        if self.assignment_unit_ids.id_count > self.record_ids.id_count:
            raise ValueError("excluded assignment-unit count cannot exceed excluded records")
        return self


class PartitionUniverse(BenchmarkModel):
    physical_dataset_binding_id: str = Field(min_length=1)
    slice_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    assignment_unit: ExperimentalUnitBinding
    assignment_unit_ids: CanonicalIdMembership
    record_unit: ExperimentalUnitBinding
    record_ids: CanonicalIdMembership
    descendant_closure_artifact: ContentAddressedArtifact
    exclusions: ExcludedPartitionMembership | None = None

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @field_validator("slice_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def universe_is_coherent(self) -> PartitionUniverse:
        _canonical_id(
            self.physical_dataset_binding_id,
            name="partition-universe physical dataset binding",
        )
        if self.assignment_unit_ids.id_count > self.record_ids.id_count:
            raise ValueError("universe assignment-unit count cannot exceed record count")
        return self


class BenchmarkPartitionRole(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    MODEL_SELECTION_VALIDATION = "model_selection_validation"
    UNTOUCHED_TEST = "untouched_test"
    EXTERNAL_VALIDATION = "external_validation"


class BenchmarkPartition(BenchmarkModel):
    partition_id: str = Field(min_length=1)
    role: BenchmarkPartitionRole
    physical_dataset_binding_id: str = Field(min_length=1)
    membership: PartitionMembershipSpec

    @field_validator("partition_id", "physical_dataset_binding_id")
    @classmethod
    def ids_are_canonical(cls, value: str) -> str:
        return _canonical_id(value, name="benchmark partition identifier")

    @property
    def materialized_membership(self) -> ExplicitPartitionMembership | None:
        if isinstance(self.membership, ExplicitPartitionMembership):
            return self.membership
        return self.membership.materialized_membership


class BenchmarkSplitPlan(BenchmarkModel):
    schema_version: SplitPlanSchemaVersion = SPLIT_PLAN_SCHEMA_VERSION
    split_id: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    universes: tuple[PartitionUniverse, ...] = Field(min_length=1)
    protected_group_closures: tuple[ProtectedGroupClosure, ...] = Field(min_length=1)
    partitions: tuple[BenchmarkPartition, ...] = Field(min_length=1)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @field_validator("query_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def plan_is_content_bound_and_group_safe(self) -> BenchmarkSplitPlan:
        _canonical_id(self.split_id, name="benchmark split ID")
        _canonical_id(self.split_version, name="benchmark split version")
        universe_ids = tuple(item.physical_dataset_binding_id for item in self.universes)
        closure_ids = tuple(
            item.physical_dataset_binding_id for item in self.protected_group_closures
        )
        _unique_sorted(
            universe_ids,
            name="partition-universe physical dataset IDs",
            allow_empty=False,
        )
        _unique_sorted(
            closure_ids,
            name="protected-closure physical dataset IDs",
            allow_empty=False,
        )
        partition_ids = tuple(partition.partition_id for partition in self.partitions)
        _unique_sorted(partition_ids, name="benchmark partition IDs", allow_empty=False)
        used_datasets = {partition.physical_dataset_binding_id for partition in self.partitions}
        if set(universe_ids) != used_datasets or set(closure_ids) != used_datasets:
            raise ValueError(
                "partitions, universes, and protected closures must cover exact physical datasets"
            )
        universes = {item.physical_dataset_binding_id: item for item in self.universes}
        closures = {
            item.physical_dataset_binding_id: item for item in self.protected_group_closures
        }
        for partition in self.partitions:
            universe = universes[partition.physical_dataset_binding_id]
            closure = closures[partition.physical_dataset_binding_id]
            if (
                universe.assignment_unit != closure.assignment_unit
                or universe.record_unit != closure.record_unit
            ):
                raise ValueError("partition universe units must match the protected closure")
            membership = partition.materialized_membership
            declared_assignment = partition.membership.assignment_unit
            if declared_assignment != closure.assignment_unit:
                raise ValueError("partition assignment unit must match its protected closure")
            if (
                isinstance(partition.membership, PartitionGenerationSpec)
                and partition.membership.source_universe_fingerprint != universe.fingerprint
            ):
                raise ValueError("partition generator must bind the exact source universe")
            if membership is None:
                continue
            if membership.record_unit != closure.record_unit:
                raise ValueError("partition record unit must match its protected closure")
            expected_groups = tuple(group.unit.key for group in closure.protected_groups)
            actual_groups = tuple(item.unit.key for item in membership.protected_group_memberships)
            if actual_groups != expected_groups:
                raise ValueError("partition must bind membership for every protected group")
        for physical_dataset_id, universe in universes.items():
            materialized = tuple(
                partition.materialized_membership
                for partition in self.partitions
                if partition.physical_dataset_binding_id == physical_dataset_id
            )
            if any(item is None for item in materialized):
                continue
            outputs = tuple(item for item in materialized if item is not None)
            excluded_records = universe.exclusions.record_ids.id_count if universe.exclusions else 0
            excluded_units = (
                universe.exclusions.assignment_unit_ids.id_count if universe.exclusions else 0
            )
            if sum(item.record_ids.id_count for item in outputs) + excluded_records != (
                universe.record_ids.id_count
            ):
                raise ValueError("partition records plus exclusions must exhaust the universe")
            if sum(item.assignment_unit_ids.id_count for item in outputs) + excluded_units != (
                universe.assignment_unit_ids.id_count
            ):
                raise ValueError("partition units plus exclusions must exhaust the universe")
        return self


class MetricDirection(StrEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class PredictionRepresentation(StrEnum):
    POSTERIOR_SAMPLES = "posterior_samples"
    DISTRIBUTION_PARAMETERS = "distribution_parameters"
    PROBABILITIES = "probabilities"
    POINT_ESTIMATE = "point_estimate"


class TargetRepresentation(StrEnum):
    COUNT_MATRIX = "count_matrix"
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    EVENT_TIME = "event_time"
    EMPIRICAL_DISTRIBUTION = "empirical_distribution"


class MetricAggregation(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"
    POOLED_RECORD = "pooled_record"


class MetricWeightingScheme(StrEnum):
    EQUAL_EVALUATION_UNIT = "equal_evaluation_unit"
    EQUAL_GROUP_THEN_EQUAL_EVALUATION_UNIT = "equal_group_then_equal_evaluation_unit"
    CONTENT_ADDRESSED_FIXED = "content_addressed_fixed"
    RECORD_COUNT_WEIGHTED = "record_count_weighted"


class MetricWeightingPolicy(BenchmarkModel):
    scheme: MetricWeightingScheme
    group_dependence_id: str | None = Field(default=None, min_length=1)
    fixed_weights_artifact: ContentAddressedArtifact | None = None
    forbid_implicit_record_count_weighting: Literal[True] = True

    @model_validator(mode="after")
    def weighting_is_explicit(self) -> MetricWeightingPolicy:
        if self.group_dependence_id is not None:
            _canonical_id(self.group_dependence_id, name="metric weighting group")
        if self.scheme is MetricWeightingScheme.EQUAL_GROUP_THEN_EQUAL_EVALUATION_UNIT:
            if self.group_dependence_id is None or self.fixed_weights_artifact is not None:
                raise ValueError("equal-group weighting requires exactly one declared group")
        elif self.scheme is MetricWeightingScheme.CONTENT_ADDRESSED_FIXED:
            if self.fixed_weights_artifact is None or self.group_dependence_id is not None:
                raise ValueError("fixed weighting requires only a content-addressed weight table")
        elif self.group_dependence_id is not None or self.fixed_weights_artifact is not None:
            raise ValueError("un-grouped weighting must omit group and fixed-weight artifacts")
        return self


class MetricMissingnessPolicy(StrEnum):
    ERROR_ON_MISSING = "error_on_missing"
    COMPLETE_INDEPENDENT_UNITS = "complete_independent_units"
    MASK_WITH_REPORTED_DENOMINATOR = "mask_with_reported_denominator"


class MetricResamplingScheme(StrEnum):
    IID_EVALUATION_UNIT = "iid_evaluation_unit"
    CLUSTERED = "clustered"
    MULTIWAY_CLUSTERED = "multiway_clustered"
    BLOCKED_PERMUTATION = "blocked_permutation"


class MetricDependenceKind(StrEnum):
    EXPERIMENTAL_UNIT = "experimental_unit"
    INTERVENTION_CONDITION = "intervention_condition"
    ENVIRONMENT_CONDITION = "environment_condition"
    SUBJECT_GROUP = "subject_group"


class MetricDependenceUnit(BenchmarkModel):
    dependence_id: str = Field(min_length=1)
    kind: MetricDependenceKind
    identity: UnitIdentityExpression
    record_to_group_artifact: ContentAddressedArtifact
    experimental_unit: ExperimentalUnitBinding | None = None
    parent_dependence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def dependence_unit_is_exact(self) -> MetricDependenceUnit:
        _canonical_id(self.dependence_id, name="metric dependence-unit ID")
        _unique_sorted(self.parent_dependence_ids, name="metric dependence parents")
        if (self.kind is MetricDependenceKind.EXPERIMENTAL_UNIT) is (
            self.experimental_unit is None
        ):
            raise ValueError(
                "experimental dependence units require, and only they permit, a unit binding"
            )
        if self.experimental_unit is not None and self.experimental_unit.identity != self.identity:
            raise ValueError("dependence identity must match its experimental unit")
        return self


class ExecutableImplementationBinding(BenchmarkModel):
    kind: Literal["executable"] = "executable"
    specification_artifact: ContentAddressedArtifact
    implementation: VersionedImplementation
    golden_fixture_artifact: ContentAddressedArtifact


class SpecificationOnlyImplementationBinding(BenchmarkModel):
    kind: Literal["specification_only"] = "specification_only"
    specification_artifact: ContentAddressedArtifact
    blockers: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def blockers_are_canonical(self) -> SpecificationOnlyImplementationBinding:
        _unique_sorted(
            self.blockers,
            name="specification-only implementation blockers",
            allow_empty=False,
        )
        return self


BenchmarkImplementationBinding: TypeAlias = Annotated[
    ExecutableImplementationBinding | SpecificationOnlyImplementationBinding,
    Field(discriminator="kind"),
]


class MetricUncertaintySpec(BenchmarkModel):
    method: BenchmarkImplementationBinding
    resampling_scheme: MetricResamplingScheme
    dependence_units: tuple[MetricDependenceUnit, ...] = Field(min_length=1)
    confidence_level: float = Field(gt=0, lt=1)
    resample_count: int = Field(gt=0)

    @field_validator("confidence_level")
    @classmethod
    def confidence_is_finite(cls, value: float) -> float:
        return require_finite(value, name="metric confidence level")

    @model_validator(mode="after")
    def dependence_structure_is_explicit(self) -> MetricUncertaintySpec:
        dependence_ids = tuple(item.dependence_id for item in self.dependence_units)
        _unique_sorted(dependence_ids, name="metric dependence-unit IDs", allow_empty=False)
        known = set(dependence_ids)
        parents_by_id = {
            item.dependence_id: set(item.parent_dependence_ids) for item in self.dependence_units
        }
        for dependence_id, parents in parents_by_id.items():
            if dependence_id in parents or not parents <= known:
                raise ValueError("metric dependence parents must be known and non-reflexive")

        pending = {key: set(value) for key, value in parents_by_id.items()}
        while pending:
            roots = {key for key, parents in pending.items() if not parents}
            if not roots:
                raise ValueError("metric dependence hierarchy must be acyclic")
            pending = {key: parents - roots for key, parents in pending.items() if key not in roots}

        if (
            self.resampling_scheme is MetricResamplingScheme.IID_EVALUATION_UNIT
            and len(self.dependence_units) != 1
        ):
            raise ValueError("iid uncertainty requires exactly one evaluation-unit grouping")
        if (
            self.resampling_scheme is MetricResamplingScheme.MULTIWAY_CLUSTERED
            and len(self.dependence_units) < 2
        ):
            raise ValueError("multiway-clustered uncertainty requires at least two groupings")
        return self


class BenchmarkMetricDefinition(BenchmarkModel):
    schema_version: MetricDefinitionSchemaVersion = METRIC_DEFINITION_SCHEMA_VERSION
    metric_id: str = Field(min_length=1)
    metric_version: str = Field(min_length=1)
    family: MetricFamily
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    evidence_binding_ids: tuple[str, ...] = Field(min_length=1)
    evaluation_partition_ids: tuple[str, ...] = Field(min_length=1)
    target_output_keys: tuple[str, ...] = Field(min_length=1)
    horizon_names: tuple[str, ...] = Field(min_length=1)
    implementation_binding: BenchmarkImplementationBinding
    prediction_representation: PredictionRepresentation
    target_representation: TargetRepresentation
    evaluation_unit: ExperimentalUnitBinding
    aggregation: MetricAggregation
    weighting: MetricWeightingPolicy
    direction: MetricDirection
    units: str = Field(min_length=1)
    formula: str = Field(min_length=1)
    parameters: tuple[BenchmarkParameter, ...] = ()
    missingness_policy: MetricMissingnessPolicy
    minimum_coverage: float | None = Field(default=None, gt=0, le=1)
    uncertainty: MetricUncertaintySpec
    minimum_evaluation_units: int = Field(gt=0)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @field_validator("query_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @field_validator("minimum_coverage")
    @classmethod
    def coverage_is_finite(cls, value: float | None) -> float | None:
        if value is not None:
            return require_finite(value, name="minimum metric coverage")
        return value

    @model_validator(mode="after")
    def metric_is_exact(self) -> BenchmarkMetricDefinition:
        _canonical_id(self.metric_id, name="benchmark metric ID")
        _canonical_id(self.metric_version, name="benchmark metric version")
        _canonical_id(self.units, name="benchmark metric units")
        _canonical_id(self.formula, name="benchmark metric formula")
        for values, name in (
            (self.evidence_binding_ids, "metric evidence bindings"),
            (self.evaluation_partition_ids, "metric evaluation partitions"),
            (self.target_output_keys, "metric target-output keys"),
            (self.horizon_names, "metric horizon names"),
        ):
            _unique_sorted(values, name=name, allow_empty=False)
        _parameters_are_canonical(self.parameters)
        if (self.missingness_policy is MetricMissingnessPolicy.MASK_WITH_REPORTED_DENOMINATOR) is (
            self.minimum_coverage is None
        ):
            raise ValueError("masked metrics require, and only masked metrics permit, coverage")
        if self.uncertainty.resampling_scheme is MetricResamplingScheme.IID_EVALUATION_UNIT:
            dependence = self.uncertainty.dependence_units[0]
            if (
                dependence.kind is not MetricDependenceKind.EXPERIMENTAL_UNIT
                or dependence.experimental_unit != self.evaluation_unit
            ):
                raise ValueError("iid uncertainty must resample the declared evaluation unit")
        dependence_ids = {item.dependence_id for item in self.uncertainty.dependence_units}
        if (
            self.weighting.scheme is MetricWeightingScheme.EQUAL_GROUP_THEN_EQUAL_EVALUATION_UNIT
            and self.weighting.group_dependence_id not in dependence_ids
        ):
            raise ValueError("metric weighting group must resolve in the dependence structure")
        record_weighted = self.weighting.scheme is MetricWeightingScheme.RECORD_COUNT_WEIGHTED
        if (self.aggregation is MetricAggregation.POOLED_RECORD) is not record_weighted:
            raise ValueError("pooled-record aggregation requires explicit record-count weighting")
        return self


class MetricDefinitionReference(BenchmarkModel):
    metric_id: str = Field(min_length=1)
    metric_version: str = Field(min_length=1)
    metric_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    @field_validator("metric_id", "metric_version")
    @classmethod
    def ids_are_canonical(cls, value: str) -> str:
        return _canonical_id(value, name="metric reference identity")

    @field_validator("metric_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()


class BaselineApplicabilityRule(BenchmarkModel):
    allowed_subject_kinds: tuple[SubjectKind, ...] = Field(min_length=1)
    requires_intervention_space: bool = False
    requires_environment_space: bool = False
    requires_pre_cutoff_target_observation: bool = False
    minimum_target_count: int = Field(gt=0)
    minimum_horizon_count: int = Field(gt=0)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @model_validator(mode="after")
    def allowed_subjects_are_canonical(self) -> BaselineApplicabilityRule:
        values = tuple(kind.value for kind in self.allowed_subject_kinds)
        _unique_sorted(values, name="baseline allowed subject kinds", allow_empty=False)
        return self

    def applies_to(self, query: StateQuery) -> bool:
        target_keys = {output.term.key for output in query.target_outputs}
        evidence_modality_keys = {
            modality.key for modality in query.evidence_policy.allowed_modalities
        }
        has_pre_cutoff_target_observation = (
            query.evidence_policy.minimum_observed_measurements > 0
            and bool(target_keys & evidence_modality_keys)
        )
        return (
            query.subject.kind in self.allowed_subject_kinds
            and (not self.requires_intervention_space or bool(query.intervention_space))
            and (not self.requires_environment_space or bool(query.environment_space))
            and (
                not self.requires_pre_cutoff_target_observation or has_pre_cutoff_target_observation
            )
            and len(query.target_outputs) >= self.minimum_target_count
            and len(query.prediction_horizons) >= self.minimum_horizon_count
        )


class BenchmarkBaselineDefinition(BenchmarkModel):
    schema_version: BaselineDefinitionSchemaVersion = BASELINE_DEFINITION_SCHEMA_VERSION
    baseline_id: str = Field(min_length=1)
    baseline_version: str = Field(min_length=1)
    query_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    implementation_binding: BenchmarkImplementationBinding
    applicability: BaselineApplicabilityRule
    training_partition_ids: tuple[str, ...] = ()
    fixed_model_artifact: ContentAddressedArtifact | None = None
    parameters: tuple[BenchmarkParameter, ...] = ()
    seeds: tuple[int, ...] = Field(min_length=1)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @field_validator("query_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @field_validator("seeds")
    @classmethod
    def seeds_are_unique_sorted(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if len(values) != len(set(values)) or tuple(sorted(values)) != values:
            raise ValueError("baseline seeds must be unique and sorted")
        return values

    @model_validator(mode="after")
    def baseline_is_reproducible(self) -> BenchmarkBaselineDefinition:
        _canonical_id(self.baseline_id, name="benchmark baseline ID")
        _canonical_id(self.baseline_version, name="benchmark baseline version")
        _unique_sorted(self.training_partition_ids, name="baseline training partitions")
        _parameters_are_canonical(self.parameters)
        if not self.training_partition_ids and self.fixed_model_artifact is None:
            raise ValueError("a baseline requires training partitions or a fixed model artifact")
        return self


class BaselineDefinitionReference(BenchmarkModel):
    baseline_id: str = Field(min_length=1)
    baseline_version: str = Field(min_length=1)
    baseline_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    @field_validator("baseline_id", "baseline_version")
    @classmethod
    def ids_are_canonical(cls, value: str) -> str:
        return _canonical_id(value, name="baseline reference identity")

    @field_validator("baseline_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()


class LeakageCheckKind(StrEnum):
    RECORD_MEMBERSHIP_DISJOINT = "record_membership_disjoint"
    PROTECTED_GROUP_DISJOINT = "protected_group_disjoint"
    SOURCE_DUPLICATE_DISJOINT = "source_duplicate_disjoint"
    PREPROCESSING_FIT_ISOLATED = "preprocessing_fit_isolated"
    TARGET_DERIVATION_ISOLATED = "target_derivation_isolated"
    TEMPORAL_CUTOFF_RESPECTED = "temporal_cutoff_respected"


class AuditCheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_ASSESSED = "not_assessed"


_OVERLAP_CHECKS = {
    LeakageCheckKind.RECORD_MEMBERSHIP_DISJOINT,
    LeakageCheckKind.PROTECTED_GROUP_DISJOINT,
    LeakageCheckKind.SOURCE_DUPLICATE_DISJOINT,
}


class LeakageAuditCheck(BenchmarkModel):
    check_id: str = Field(min_length=1)
    kind: LeakageCheckKind
    status: AuditCheckStatus
    physical_dataset_binding_id: str | None = Field(default=None, min_length=1)
    protected_unit: ExperimentalUnitBinding | None = None
    partition_ids: tuple[str, ...] = ()
    comparisons_expected: int | None = Field(default=None, ge=0)
    comparisons_completed: int | None = Field(default=None, ge=0)
    overlap_count: int | None = Field(default=None, ge=0)
    report_locator: str = Field(min_length=1)
    notes: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    @model_validator(mode="after")
    def result_is_fail_closed(self) -> LeakageAuditCheck:
        _canonical_id(self.check_id, name="leakage-check ID")
        _canonical_id(self.report_locator, name="leakage report locator")
        if self.physical_dataset_binding_id is not None:
            _canonical_id(
                self.physical_dataset_binding_id,
                name="leakage physical dataset binding",
            )
        _unique_sorted(self.partition_ids, name="leakage-check partitions")
        _unique_sorted(self.notes, name="leakage-check notes")
        _unique_sorted(self.blockers, name="leakage-check blockers")
        overlap_values = (
            self.comparisons_expected,
            self.comparisons_completed,
            self.overlap_count,
        )
        if self.kind in _OVERLAP_CHECKS:
            if any(value is None for value in overlap_values):
                raise ValueError("overlap checks require complete comparison and overlap counts")
            if self.comparisons_expected != self.comparisons_completed:
                raise ValueError("overlap checks must complete every declared comparison")
            if self.status is AuditCheckStatus.PASSED and self.overlap_count != 0:
                raise ValueError("passed overlap checks require zero overlap")
        elif any(value is not None for value in overlap_values):
            raise ValueError("non-overlap checks must omit overlap-count fields")
        if self.kind is LeakageCheckKind.PROTECTED_GROUP_DISJOINT:
            if self.physical_dataset_binding_id is None or self.protected_unit is None:
                raise ValueError(
                    "protected-group checks require physical dataset and unit identity"
                )
        elif self.protected_unit is not None:
            raise ValueError("only protected-group checks may declare a protected unit")
        if self.kind is LeakageCheckKind.RECORD_MEMBERSHIP_DISJOINT and (
            self.physical_dataset_binding_id is None
        ):
            raise ValueError("record-membership checks require a physical dataset binding")
        if self.status is AuditCheckStatus.PASSED:
            if not self.notes or self.blockers:
                raise ValueError("passed leakage checks require notes and no blockers")
        elif not self.blockers:
            raise ValueError("failed or unassessed leakage checks require blockers")
        return self


class BenchmarkLeakageAudit(BenchmarkModel):
    schema_version: LeakageAuditSchemaVersion = LEAKAGE_AUDIT_SCHEMA_VERSION
    audit_id: str = Field(min_length=1)
    audit_version: str = Field(min_length=1)
    split_plan_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    implementation: VersionedImplementation
    report_artifact: ContentAddressedArtifact
    evaluated_partition_ids: tuple[str, ...] = Field(min_length=1)
    checks: tuple[LeakageAuditCheck, ...] = Field(min_length=1)
    reviewed_by: tuple[str, ...] = Field(min_length=1)
    reviewed_on: date

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @field_validator("split_plan_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def audit_is_canonical(self) -> BenchmarkLeakageAudit:
        _canonical_id(self.audit_id, name="leakage-audit ID")
        _canonical_id(self.audit_version, name="leakage-audit version")
        _unique_sorted(
            self.evaluated_partition_ids,
            name="audited partition IDs",
            allow_empty=False,
        )
        check_ids = tuple(check.check_id for check in self.checks)
        _unique_sorted(check_ids, name="leakage-check IDs", allow_empty=False)
        _unique_sorted(self.reviewed_by, name="leakage reviewers", allow_empty=False)
        return self


class ThresholdComparison(StrEnum):
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"


class ThresholdEstimate(StrEnum):
    POINT = "point"
    LOWER_CONFIDENCE_BOUND = "lower_confidence_bound"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"


class BaselineMarginMode(StrEnum):
    ABSOLUTE_DIFFERENCE = "absolute_difference"
    RELATIVE_FRACTION = "relative_fraction"


class BaselineRequirement(StrEnum):
    NONINFERIOR = "noninferior"
    SUPERIOR = "superior"


class ExactBaselineComparator(BenchmarkModel):
    kind: Literal["exact_baseline"] = "exact_baseline"
    baseline: BaselineDefinitionReference


class BestApplicableBaselineComparator(BenchmarkModel):
    kind: Literal["best_applicable_baseline"] = "best_applicable_baseline"
    baselines: tuple[BaselineDefinitionReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def baseline_pool_is_canonical(self) -> BestApplicableBaselineComparator:
        identities = tuple(item.baseline_id for item in self.baselines)
        _unique_sorted(identities, name="best-applicable baseline IDs", allow_empty=False)
        return self


BaselineComparator: TypeAlias = Annotated[
    ExactBaselineComparator | BestApplicableBaselineComparator,
    Field(discriminator="kind"),
]


class BenchmarkAcceptanceRule(BenchmarkModel):
    rule_id: str = Field(min_length=1)
    metric: MetricDefinitionReference
    partition_id: str = Field(min_length=1)
    comparison: ThresholdComparison
    estimate: ThresholdEstimate
    absolute_threshold: float | None = None
    baseline_margin: float | None = None
    baseline_comparator: BaselineComparator | None = None
    baseline_margin_mode: BaselineMarginMode | None = None
    baseline_requirement: BaselineRequirement | None = None
    confidence_level: float | None = Field(default=None, gt=0, lt=1)
    rationale: str = Field(min_length=1)

    @field_validator("absolute_threshold", "baseline_margin", "confidence_level")
    @classmethod
    def numeric_values_are_finite(cls, value: float | None) -> float | None:
        if value is not None:
            return require_finite(value, name="benchmark acceptance rule")
        return value

    @model_validator(mode="after")
    def threshold_is_complete(self) -> BenchmarkAcceptanceRule:
        _canonical_id(self.rule_id, name="acceptance-rule ID")
        _canonical_id(self.partition_id, name="acceptance-rule partition ID")
        _canonical_id(self.rationale, name="acceptance-rule rationale")
        if (self.absolute_threshold is None) is (self.baseline_margin is None):
            raise ValueError("acceptance rule requires exactly one absolute or baseline threshold")
        baseline_fields = (
            self.baseline_comparator,
            self.baseline_margin_mode,
            self.baseline_requirement,
        )
        if self.baseline_margin is None:
            if any(value is not None for value in baseline_fields):
                raise ValueError("absolute thresholds must omit all baseline semantics")
        elif any(value is None for value in baseline_fields):
            raise ValueError("baseline margins require an exact baseline, mode, and requirement")
        elif self.baseline_margin < 0:
            raise ValueError("baseline margins must be nonnegative magnitudes")
        elif (
            self.baseline_margin_mode is BaselineMarginMode.RELATIVE_FRACTION
            and self.baseline_margin > 1
        ):
            raise ValueError("relative baseline margins must lie in [0, 1]")
        if self.baseline_margin is not None and self.estimate is ThresholdEstimate.POINT:
            raise ValueError("baseline-relative rules require a paired one-sided confidence bound")
        uses_confidence = self.estimate is not ThresholdEstimate.POINT
        if uses_confidence is (self.confidence_level is None):
            raise ValueError("confidence-bound rules require, and point rules omit, confidence")
        return self


class AcceptanceGroupOperator(StrEnum):
    ALL = "all"
    ANY = "any"


class BenchmarkAcceptanceGroup(BenchmarkModel):
    group_id: str = Field(min_length=1)
    operator: AcceptanceGroupOperator
    rule_ids: tuple[str, ...] = ()
    child_group_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def group_is_canonical(self) -> BenchmarkAcceptanceGroup:
        _canonical_id(self.group_id, name="acceptance-group ID")
        _unique_sorted(self.rule_ids, name="acceptance-group rule IDs")
        _unique_sorted(self.child_group_ids, name="acceptance child-group IDs")
        if not self.rule_ids and not self.child_group_ids:
            raise ValueError("acceptance groups require at least one rule or child group")
        if self.group_id in self.child_group_ids:
            raise ValueError("acceptance groups cannot contain themselves")
        return self


class BenchmarkAcceptancePolicy(BenchmarkModel):
    policy_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    root_group_id: str = Field(min_length=1)
    groups: tuple[BenchmarkAcceptanceGroup, ...] = Field(min_length=1)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @model_validator(mode="after")
    def policy_is_one_complete_tree(self) -> BenchmarkAcceptancePolicy:
        _canonical_id(self.policy_id, name="acceptance-policy ID")
        _canonical_id(self.policy_version, name="acceptance-policy version")
        _canonical_id(self.root_group_id, name="acceptance root-group ID")
        group_ids = tuple(group.group_id for group in self.groups)
        _unique_sorted(group_ids, name="acceptance-policy group IDs", allow_empty=False)
        groups = {group.group_id: group for group in self.groups}
        if self.root_group_id not in groups:
            raise ValueError("acceptance-policy root group must exist")
        parent_counts = {group_id: 0 for group_id in group_ids}
        for group in self.groups:
            for child_id in group.child_group_ids:
                if child_id not in groups:
                    raise ValueError("acceptance policy references an unknown child group")
                parent_counts[child_id] += 1
        if parent_counts[self.root_group_id] != 0 or any(
            count != 1
            for group_id, count in parent_counts.items()
            if group_id != self.root_group_id
        ):
            raise ValueError("acceptance groups must form one rooted tree")
        pending = {self.root_group_id}
        visited: set[str] = set()
        while pending:
            group_id = pending.pop()
            if group_id in visited:
                raise ValueError("acceptance group tree must be acyclic")
            visited.add(group_id)
            pending.update(groups[group_id].child_group_ids)
        if visited != set(group_ids):
            raise ValueError("every acceptance group must be reachable from the root")
        rule_ids = tuple(rule_id for group in self.groups for rule_id in group.rule_ids)
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("each acceptance rule must occur in exactly one group")
        return self


class MetricResult(BenchmarkModel):
    metric: MetricDefinitionReference
    partition_id: str = Field(min_length=1)
    value: float
    lower_confidence_bound: float | None = None
    upper_confidence_bound: float | None = None
    evaluated_evaluation_units: int = Field(gt=0)
    evaluated_case_ids: CanonicalIdMembership
    result_artifact: ContentAddressedArtifact

    @field_validator("value", "lower_confidence_bound", "upper_confidence_bound")
    @classmethod
    def values_are_finite(cls, value: float | None) -> float | None:
        if value is not None:
            return require_finite(value, name="metric result")
        return value

    @model_validator(mode="after")
    def interval_is_ordered(self) -> MetricResult:
        _canonical_id(self.partition_id, name="metric-result partition ID")
        bounds = (self.lower_confidence_bound, self.upper_confidence_bound)
        if (bounds[0] is None) != (bounds[1] is None):
            raise ValueError("metric-result confidence bounds must be supplied together")
        if (
            bounds[0] is not None
            and bounds[1] is not None
            and not bounds[0] <= self.value <= bounds[1]
        ):
            raise ValueError("metric result must lie inside its confidence interval")
        return self


class PairedConfidenceBoundKind(StrEnum):
    LOWER = "lower"
    UPPER = "upper"


class PairedBaselineComparisonResult(BenchmarkModel):
    comparison_id: str = Field(min_length=1)
    metric: MetricDefinitionReference
    partition_id: str = Field(min_length=1)
    baseline: BaselineDefinitionReference
    effect_scale: BaselineMarginMode
    effect_definition: Literal[
        "candidate_minus_baseline_v1",
        "candidate_minus_baseline_over_abs_baseline_v1",
    ]
    point_effect: float
    one_sided_confidence_bound: float
    bound_kind: PairedConfidenceBoundKind
    confidence_level: float = Field(gt=0, lt=1)
    evaluated_case_ids: CanonicalIdMembership
    dependence_ids: tuple[str, ...] = Field(min_length=1)
    paired_block_membership_artifact: ContentAddressedArtifact
    result_artifact: ContentAddressedArtifact

    @field_validator(
        "point_effect",
        "one_sided_confidence_bound",
        "confidence_level",
    )
    @classmethod
    def numeric_values_are_finite(cls, value: float) -> float:
        return require_finite(value, name="paired baseline comparison")

    @model_validator(mode="after")
    def comparison_is_exact(self) -> PairedBaselineComparisonResult:
        _canonical_id(self.comparison_id, name="paired-comparison ID")
        _canonical_id(self.partition_id, name="paired-comparison partition ID")
        _unique_sorted(
            self.dependence_ids,
            name="paired-comparison dependence IDs",
            allow_empty=False,
        )
        expected_definition = (
            "candidate_minus_baseline_v1"
            if self.effect_scale is BaselineMarginMode.ABSOLUTE_DIFFERENCE
            else "candidate_minus_baseline_over_abs_baseline_v1"
        )
        if self.effect_definition != expected_definition:
            raise ValueError("paired comparison definition must match its effect scale")
        if self.bound_kind is PairedConfidenceBoundKind.LOWER:
            if self.one_sided_confidence_bound > self.point_effect:
                raise ValueError("lower paired bound cannot exceed the point effect")
        elif self.one_sided_confidence_bound < self.point_effect:
            raise ValueError("upper paired bound cannot be below the point effect")
        return self


class BaselineRunStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"
    CRASHED = "crashed"
    NOT_RUN = "not_run"


class BaselineRun(BenchmarkModel):
    baseline: BaselineDefinitionReference
    status: BaselineRunStatus
    applicability_rule_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    prediction_artifact: ContentAddressedArtifact | None = None
    metric_results: tuple[MetricResult, ...] = ()
    notes: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    @field_validator("applicability_rule_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def run_is_fail_closed(self) -> BaselineRun:
        _unique_sorted(self.notes, name="baseline-run notes")
        _unique_sorted(self.blockers, name="baseline-run blockers")
        metric_keys = tuple(
            f"{result.metric.metric_id}:{result.partition_id}" for result in self.metric_results
        )
        _unique_sorted(metric_keys, name="baseline metric-result keys")
        if self.status is BaselineRunStatus.PASSED:
            if self.prediction_artifact is None or not self.metric_results or self.blockers:
                raise ValueError(
                    "passed baselines require predictions, metric results, no blockers"
                )
        elif self.status is BaselineRunStatus.NOT_APPLICABLE:
            if self.prediction_artifact is not None or self.metric_results or self.blockers:
                raise ValueError("not-applicable baselines must omit outputs and blockers")
            if not self.notes:
                raise ValueError("not-applicable baselines require a derived explanation")
        elif not self.blockers:
            raise ValueError("failed, crashed, or unrun baselines require blockers")
        return self


class BenchmarkLifecycle(StrEnum):
    DRAFT = "draft"
    FROZEN = "frozen"
    SUPERSEDED = "superseded"


class BenchmarkIntent(StrEnum):
    SCIENTIFIC = "scientific"
    COMPONENT_BENCHMARK = "component_benchmark"
    TECHNICAL_FIXTURE = "technical_fixture"


class BenchmarkAdmissionStatus(StrEnum):
    BLOCKED = "blocked"
    COMPONENT_BENCHMARK = "component_benchmark"
    TECHNICAL_ONLY = "technical_only"
    ADMITTED = "admitted"


class BenchmarkDefinition(BenchmarkModel):
    schema_version: BenchmarkSchemaVersion = BENCHMARK_SCHEMA_VERSION
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    design_status: BenchmarkLifecycle
    supersedes_version: str | None = Field(default=None, min_length=1)
    intent: BenchmarkIntent
    query: StateQueryBinding
    scope: BenchmarkScope
    evidence_bindings: tuple[BenchmarkEvidenceBinding, ...] = Field(min_length=1)
    split_plan: BenchmarkSplitPlan | None = None
    evaluation_case_set: BenchmarkEvaluationCaseSet | None = None
    metrics: tuple[BenchmarkMetricDefinition, ...] = ()
    baselines: tuple[BenchmarkBaselineDefinition, ...] = ()
    acceptance_rules: tuple[BenchmarkAcceptanceRule, ...] = ()
    acceptance_policy: BenchmarkAcceptancePolicy | None = None
    notes: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @model_validator(mode="after")
    def definition_is_exact_and_scoped(self) -> BenchmarkDefinition:
        _canonical_id(self.benchmark_id, name="benchmark ID")
        _canonical_id(self.benchmark_version, name="benchmark version")
        if (self.design_status is BenchmarkLifecycle.SUPERSEDED) is (
            self.supersedes_version is None
        ):
            raise ValueError("only superseded benchmarks declare the replacement version")
        query = self.query.state_query
        scope = self.scope
        if scope.query_fingerprint != self.query.query_fingerprint:
            raise ValueError("benchmark scope must bind the exact StateQuery")
        if scope.subject_kind is not query.subject.kind:
            raise ValueError("benchmark subject kind must exactly project the StateQuery")
        if scope.system_boundary is not query.system_boundary:
            raise ValueError("benchmark system boundary must exactly project the StateQuery")
        if scope.biological_system.key != query.subject.biological_system.key:
            raise ValueError("benchmark biological system must exactly project the StateQuery")
        if scope.target_output_keys != tuple(
            sorted(output.term.key for output in query.target_outputs)
        ):
            raise ValueError("benchmark targets must exactly project the StateQuery")
        if scope.horizon_names != tuple(sorted(item.name for item in query.prediction_horizons)):
            raise ValueError("benchmark horizons must exactly project the StateQuery")
        if scope.intervention_spec_ids != tuple(
            sorted(item.spec_id for item in query.intervention_space)
        ):
            raise ValueError("benchmark interventions must exactly project the StateQuery")
        if (
            self.intent
            in {
                BenchmarkIntent.SCIENTIFIC,
                BenchmarkIntent.COMPONENT_BENCHMARK,
            }
            and not scope.scientific_claims
        ):
            raise ValueError("scientific and component benchmarks require explicit claims")
        if self.intent is BenchmarkIntent.TECHNICAL_FIXTURE and scope.scientific_claims:
            raise ValueError("technical fixtures must not assert biological performance claims")

        evidence_ids = tuple(item.binding_id for item in self.evidence_bindings)
        _unique_sorted(evidence_ids, name="benchmark evidence-binding IDs", allow_empty=False)
        known_evidence = set(evidence_ids)
        evidence_by_id = {item.binding_id: item for item in self.evidence_bindings}
        physical_datasets: dict[str, BenchmarkEvidenceBinding] = {}
        for binding in self.evidence_bindings:
            existing = physical_datasets.setdefault(
                binding.physical_dataset_binding_id,
                binding,
            )
            if (
                existing.dataset_id != binding.dataset_id
                or existing.dataset_version != binding.dataset_version
                or existing.manifest_fingerprint != binding.manifest_fingerprint
                or existing.manifest_artifact.sha256 != binding.manifest_artifact.sha256
                or existing.manifest_artifact.byte_count != binding.manifest_artifact.byte_count
                or existing.manifest_artifact.media_type != binding.manifest_artifact.media_type
            ):
                raise ValueError(
                    "one physical dataset binding must identify one exact manifest artifact"
                )
        metric_ids = tuple(metric.metric_id for metric in self.metrics)
        _unique_sorted(metric_ids, name="benchmark metric IDs")
        baseline_ids = tuple(baseline.baseline_id for baseline in self.baselines)
        _unique_sorted(baseline_ids, name="benchmark baseline IDs")
        rule_ids = tuple(rule.rule_id for rule in self.acceptance_rules)
        _unique_sorted(rule_ids, name="benchmark acceptance-rule IDs")
        _unique_sorted(self.notes, name="benchmark notes")

        if self.split_plan is not None:
            if self.split_plan.query_fingerprint != self.query.query_fingerprint:
                raise ValueError("split plan must bind the exact StateQuery")
            split_datasets = {
                partition.physical_dataset_binding_id for partition in self.split_plan.partitions
            }
            if not split_datasets <= set(physical_datasets):
                raise ValueError("split plan references unknown physical dataset bindings")
        elif (
            self.evaluation_case_set
            or self.metrics
            or self.baselines
            or self.acceptance_rules
            or self.acceptance_policy
        ):
            raise ValueError(
                "evaluation cases, metrics, baselines, and thresholds require a split plan"
            )

        metrics = {metric.metric_id: metric for metric in self.metrics}
        query_targets = {output.term.key: output for output in query.target_outputs}
        query_interventions = {item.spec_id: item for item in query.intervention_space}
        query_environments = {item.variable.key: item for item in query.environment_space}
        evidence_claims: set[ScientificClaim] = set()
        for binding in self.evidence_bindings:
            scope_binding = binding.scope_binding
            target_mappings = {
                item.target_output_key: item for item in scope_binding.target_mappings
            }
            if set(target_mappings) != set(query_targets):
                raise ValueError("every evidence assessment must map every exact query target")
            for target_key, output in query_targets.items():
                mapping = target_mappings[target_key]
                if (
                    mapping.target_output_fingerprint != _fingerprint(output)
                    or mapping.target_units != output.units
                    or mapping.target_aggregation != output.aggregation
                ):
                    raise ValueError("target evidence mapping must bind the exact output spec")
                if output.aggregation.experimental_unit not in {
                    mapping.aggregation_unit.level.value,
                    *mapping.aggregation_unit.identity.source_fields,
                }:
                    raise ValueError(
                        "target aggregation unit must resolve to its declared dataset unit"
                    )
            intervention_mappings = {
                item.intervention_spec_id: item for item in scope_binding.intervention_mappings
            }
            if set(intervention_mappings) != set(query_interventions):
                raise ValueError("every evidence assessment must map every exact intervention spec")
            for spec_id, intervention_spec in query_interventions.items():
                intervention_mapping = intervention_mappings[spec_id]
                if (
                    intervention_mapping.intervention_spec_fingerprint
                    != _fingerprint(intervention_spec)
                    or intervention_mapping.assessment_intervention_kind_key
                    != intervention_spec.kind.key
                ):
                    raise ValueError(
                        "intervention evidence mapping must bind the exact action domain"
                    )
            environment_mappings = {
                item.environment_variable_key: item for item in scope_binding.environment_mappings
            }
            if set(environment_mappings) != set(query_environments):
                raise ValueError(
                    "every evidence assessment must map every exact environment domain"
                )
            for variable_key, environment_spec in query_environments.items():
                environment_mapping = environment_mappings[variable_key]
                if (
                    environment_mapping.environment_spec_fingerprint
                    != _fingerprint(environment_spec)
                    or environment_mapping.assessment_environment_variable_key != variable_key
                ):
                    raise ValueError(
                        "environment evidence mapping must bind the exact variable domain"
                    )
            if scope_binding.horizon_names != scope.horizon_names:
                raise ValueError("every evidence assessment must map every exact query horizon")
            evidence_claims.update(scope_binding.scientific_claims)
        if self.intent is not BenchmarkIntent.TECHNICAL_FIXTURE and evidence_claims != set(
            scope.scientific_claims
        ):
            raise ValueError("evidence claim classes must exactly cover benchmark claims")

        case_set = self.evaluation_case_set
        if self.metrics and case_set is None:
            raise ValueError("metric definitions require an authoritative evaluation case set")
        if case_set is not None:
            if case_set.query_fingerprint != self.query.query_fingerprint:
                raise ValueError("evaluation cases must bind the exact StateQuery")
            assert self.split_plan is not None
            partitions = {
                partition.partition_id: partition for partition in self.split_plan.partitions
            }
            case_partition_ids = {
                binding.partition_id for binding in case_set.partition_memberships
            }
            if case_partition_ids != set(partitions):
                raise ValueError("evaluation cases must cover every split partition exactly")
            declared_action_ids = {
                item.intervention_spec_id for item in case_set.intervention_case_counts
            }
            if declared_action_ids != set(query_interventions):
                raise ValueError("evaluation cases must exactly cover the query action domain")
            if query_interventions and case_set.no_action_control_case_count == 0:
                raise ValueError("interventional evaluation requires explicit no-action controls")
            if {case.horizon_name for case in case_set.cases} != set(scope.horizon_names):
                raise ValueError("evaluation cases must exactly cover all query horizons")
            if any(case.target_output_keys != scope.target_output_keys for case in case_set.cases):
                raise ValueError("every evaluation case must require every benchmark target")
            closures = {
                closure.physical_dataset_binding_id: closure
                for closure in self.split_plan.protected_group_closures
            }
            for case_binding in case_set.partition_memberships:
                partition = partitions[case_binding.partition_id]
                membership = partition.materialized_membership
                if membership is None:
                    raise ValueError("evaluation cases require materialized partition membership")
                closure = closures[partition.physical_dataset_binding_id]
                if case_set.evaluation_unit not in closure.unit_ancestry:
                    raise ValueError("case evaluation unit must occur in partition ancestry")
                protected = next(
                    (
                        item
                        for item in membership.protected_group_memberships
                        if item.unit == case_set.evaluation_unit
                    ),
                    None,
                )
                if protected is None or (protected.membership != case_binding.evaluation_unit_ids):
                    raise ValueError(
                        "evaluation cases must bind exact partition evaluation-unit membership"
                    )

        closures_by_dataset = (
            {
                closure.physical_dataset_binding_id: closure
                for closure in self.split_plan.protected_group_closures
            }
            if self.split_plan is not None
            else {}
        )
        partitions_by_id = (
            {partition.partition_id: partition for partition in self.split_plan.partitions}
            if self.split_plan is not None
            else {}
        )
        partition_roles = {
            partition_id: partition.role for partition_id, partition in partitions_by_id.items()
        }
        for binding in self.evidence_bindings:
            shared_closure = closures_by_dataset.get(binding.physical_dataset_binding_id)
            if shared_closure is None:
                continue
            if binding.required_split_unit not in shared_closure.unit_ancestry:
                raise ValueError(
                    "every assessment leakage boundary must occur in its shared physical split"
                )
            required_group = next(
                (
                    group
                    for group in shared_closure.protected_groups
                    if group.unit == binding.required_split_unit
                ),
                None,
            )
            if required_group is None or (
                ProtectedGroupReason.OBJECTIVE_REQUIRED_SPLIT not in required_group.reasons
            ):
                raise ValueError(
                    "shared physical split must protect every assessment leakage boundary"
                )
        for metric in self.metrics:
            if metric.query_fingerprint != self.query.query_fingerprint:
                raise ValueError("metrics must bind the exact StateQuery")
            assert case_set is not None
            if metric.evaluation_unit != case_set.evaluation_unit:
                raise ValueError("metrics must score the authoritative case evaluation unit")
            if metric.missingness_policy is not MetricMissingnessPolicy.ERROR_ON_MISSING:
                raise ValueError("authoritative evaluation cases require error-on-missing metrics")
            if not set(metric.evidence_binding_ids) <= known_evidence:
                raise ValueError("metric references unknown evidence bindings")
            matching_metric_identities = tuple(
                identity
                for binding_id in metric.evidence_binding_ids
                for identity in (evidence_by_id[binding_id].assessment_identity,)
                if isinstance(identity, MetricAssessmentIdentity)
                and identity.metric_id == metric.metric_id
                and identity.metric_family is metric.family
            )
            if not matching_metric_identities:
                raise ValueError("each metric must match an exact assessment metric ID and family")
            if unknown_partitions := set(metric.evaluation_partition_ids) - set(partitions_by_id):
                raise ValueError(
                    f"metric references unknown evaluation partitions: {sorted(unknown_partitions)}"
                )
            metric_physical_datasets = {
                evidence_by_id[binding_id].physical_dataset_binding_id
                for binding_id in metric.evidence_binding_ids
            }
            if any(
                partitions_by_id[partition_id].physical_dataset_binding_id
                not in metric_physical_datasets
                for partition_id in metric.evaluation_partition_ids
            ):
                raise ValueError(
                    "metric evaluation partitions must belong to its exact evidence dataset"
                )
            expected_roles = {
                BenchmarkPartitionRole(identity.partition_purpose.value)
                for identity in matching_metric_identities
            }
            actual_roles = {
                partitions_by_id[partition_id].role
                for partition_id in metric.evaluation_partition_ids
            }
            if actual_roles != expected_roles:
                raise ValueError(
                    "metric evaluation partitions must match exact assessment purposes"
                )
            metric_closures = tuple(
                closures_by_dataset[evidence_by_id[binding_id].physical_dataset_binding_id]
                for binding_id in metric.evidence_binding_ids
                if evidence_by_id[binding_id].physical_dataset_binding_id in closures_by_dataset
            )
            if len(metric_closures) != len(metric.evidence_binding_ids):
                raise ValueError("each metric evidence binding requires a protected split closure")
            for binding_id, closure in zip(
                metric.evidence_binding_ids,
                metric_closures,
                strict=True,
            ):
                if metric.evaluation_unit not in closure.unit_ancestry:
                    raise ValueError("metric evaluation unit must occur in the evidence ancestry")
                assignment_index = closure.unit_ancestry.index(closure.assignment_unit)
                evaluation_index = closure.unit_ancestry.index(metric.evaluation_unit)
                if evaluation_index < assignment_index:
                    raise ValueError(
                        "metric evaluation unit cannot be coarser than split assignment"
                    )
                if (
                    metric.evaluation_unit == closure.record_unit
                    and closure.assignment_unit != closure.record_unit
                ):
                    raise ValueError(
                        "record rows cannot be treated as replicates below a coarser split boundary"
                    )
                evaluation_group = next(
                    (
                        group
                        for group in closure.protected_groups
                        if group.unit == metric.evaluation_unit
                    ),
                    None,
                )
                if evaluation_group is None or (
                    ProtectedGroupReason.METRIC_EVALUATION not in evaluation_group.reasons
                ):
                    raise ValueError(
                        "metric evaluation units require protected, content-addressed membership"
                    )
                required_binding = evidence_by_id[binding_id]
                required_matches = tuple(
                    unit
                    for unit in closure.unit_ancestry
                    if unit == required_binding.required_split_unit
                )
                if len(required_matches) != 1:
                    raise ValueError(
                        "assessment leakage boundary must occur exactly in the split ancestry"
                    )
                experimental_dependence_units = {
                    item.experimental_unit
                    for item in metric.uncertainty.dependence_units
                    if item.experimental_unit is not None
                }
                if closure.assignment_unit not in experimental_dependence_units:
                    raise ValueError(
                        "metric uncertainty must cluster at the split assignment boundary"
                    )
                for dependence_unit in experimental_dependence_units:
                    if dependence_unit not in closure.unit_ancestry:
                        raise ValueError(
                            "metric dependence unit must occur in the evidence ancestry"
                        )
                    if closure.unit_ancestry.index(dependence_unit) > evaluation_index:
                        raise ValueError(
                            "metric uncertainty cannot treat finer records as replicate clusters"
                        )
            if not set(metric.target_output_keys) <= set(scope.target_output_keys):
                raise ValueError("metric references targets outside benchmark scope")
            if not set(metric.horizon_names) <= set(scope.horizon_names):
                raise ValueError("metric references horizons outside benchmark scope")

        if self.metrics and (
            {target for metric in self.metrics for target in metric.target_output_keys}
            != set(scope.target_output_keys)
            or {horizon for metric in self.metrics for horizon in metric.horizon_names}
            != set(scope.horizon_names)
        ):
            raise ValueError("benchmark metrics must cover every case target and horizon")

        baselines = {baseline.baseline_id: baseline for baseline in self.baselines}
        for baseline in self.baselines:
            if baseline.query_fingerprint != self.query.query_fingerprint:
                raise ValueError("baselines must bind the exact StateQuery")
            if unknown := set(baseline.training_partition_ids) - set(partition_roles):
                raise ValueError(f"baseline references unknown partitions: {sorted(unknown)}")
            if any(
                partition_roles[partition_id] is not BenchmarkPartitionRole.TRAIN
                for partition_id in baseline.training_partition_ids
            ):
                raise ValueError("baselines may fit only on training partitions")

        for rule in self.acceptance_rules:
            resolved_metric = metrics.get(rule.metric.metric_id)
            if (
                resolved_metric is None
                or resolved_metric.metric_version != rule.metric.metric_version
                or resolved_metric.fingerprint != rule.metric.metric_fingerprint
            ):
                raise ValueError("acceptance rule must bind an exact metric definition")
            if rule.partition_id not in partition_roles:
                raise ValueError("acceptance rule references an unknown partition")
            if rule.partition_id not in resolved_metric.evaluation_partition_ids:
                raise ValueError(
                    "acceptance rule partition must be frozen in its metric definition"
                )
            if partition_roles[rule.partition_id] is not BenchmarkPartitionRole.UNTOUCHED_TEST:
                raise ValueError("scientific acceptance rules must use untouched test partitions")
            allowed_comparisons = (
                {
                    ThresholdComparison.LESS_THAN,
                    ThresholdComparison.LESS_THAN_OR_EQUAL,
                }
                if resolved_metric.direction is MetricDirection.MINIMIZE
                else {
                    ThresholdComparison.GREATER_THAN,
                    ThresholdComparison.GREATER_THAN_OR_EQUAL,
                }
            )
            if rule.comparison not in allowed_comparisons:
                raise ValueError("acceptance comparison must match the metric direction")
            if rule.baseline_margin is not None:
                expected_estimate = (
                    ThresholdEstimate.UPPER_CONFIDENCE_BOUND
                    if resolved_metric.direction is MetricDirection.MINIMIZE
                    else ThresholdEstimate.LOWER_CONFIDENCE_BOUND
                )
                if rule.estimate is not expected_estimate:
                    raise ValueError(
                        "baseline-relative rules require the conservative paired bound"
                    )
            if rule.baseline_comparator is not None:
                baseline_refs = (
                    (rule.baseline_comparator.baseline,)
                    if isinstance(rule.baseline_comparator, ExactBaselineComparator)
                    else rule.baseline_comparator.baselines
                )
                for baseline_ref in baseline_refs:
                    resolved_baseline = baselines.get(baseline_ref.baseline_id)
                    if (
                        resolved_baseline is None
                        or resolved_baseline.baseline_version != baseline_ref.baseline_version
                        or resolved_baseline.fingerprint != baseline_ref.baseline_fingerprint
                    ):
                        raise ValueError("acceptance rule must bind exact baseline definitions")
        if (self.acceptance_policy is None) is bool(self.acceptance_rules):
            raise ValueError(
                "acceptance rules require, and only acceptance rules permit, a policy tree"
            )
        if self.acceptance_policy is not None:
            policy_rule_ids = {
                rule_id for group in self.acceptance_policy.groups for rule_id in group.rule_ids
            }
            if policy_rule_ids != set(rule_ids):
                raise ValueError("acceptance policy must use every exact rule exactly once")
        return self


class EvidenceResolutionBinding(BenchmarkModel):
    evidence_binding_id: str = Field(min_length=1)
    assessment_reference: DatasetAssessmentReference
    resolution_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    @field_validator("evidence_binding_id")
    @classmethod
    def binding_id_is_canonical(cls, value: str) -> str:
        return _canonical_id(value, name="evidence-resolution binding ID")

    @field_validator("resolution_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()


class BenchmarkAdmission(BenchmarkModel):
    schema_version: AdmissionSchemaVersion = ADMISSION_SCHEMA_VERSION
    admission_id: str = Field(min_length=1)
    admission_version: str = Field(min_length=1)
    definition_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    status: BenchmarkAdmissionStatus
    evidence_resolutions: tuple[EvidenceResolutionBinding, ...] = Field(min_length=1)
    leakage_audit_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[a-fA-F0-9]{64}$",
    )
    baseline_runs: tuple[BaselineRun, ...] = ()
    metric_results: tuple[MetricResult, ...] = ()
    paired_baseline_comparisons: tuple[PairedBaselineComparisonResult, ...] = ()
    reasons: tuple[str, ...] = Field(min_length=1)
    reviewed_by: tuple[str, ...] = Field(min_length=1)
    reviewed_on: date

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @field_validator("definition_fingerprint", "leakage_audit_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str | None) -> str | None:
        return value.casefold() if value is not None else None

    @model_validator(mode="after")
    def decision_is_canonical(self) -> BenchmarkAdmission:
        _canonical_id(self.admission_id, name="benchmark admission ID")
        _canonical_id(self.admission_version, name="benchmark admission version")
        resolution_ids = tuple(item.evidence_binding_id for item in self.evidence_resolutions)
        _unique_sorted(resolution_ids, name="admission evidence resolutions", allow_empty=False)
        baseline_ids = tuple(item.baseline.baseline_id for item in self.baseline_runs)
        _unique_sorted(baseline_ids, name="admission baseline runs")
        result_ids = tuple(
            f"{item.metric.metric_id}:{item.partition_id}" for item in self.metric_results
        )
        _unique_sorted(result_ids, name="admission metric-result identities")
        comparison_ids = tuple(item.comparison_id for item in self.paired_baseline_comparisons)
        _unique_sorted(comparison_ids, name="paired baseline-comparison IDs")
        comparison_keys = tuple(
            f"{item.metric.metric_id}:{item.partition_id}:{item.baseline.baseline_id}"
            for item in self.paired_baseline_comparisons
        )
        if len(comparison_keys) != len(set(comparison_keys)):
            raise ValueError("paired baseline-comparison identities must be unique")
        _unique_sorted(self.reasons, name="benchmark admission reasons", allow_empty=False)
        _unique_sorted(self.reviewed_by, name="benchmark admission reviewers", allow_empty=False)
        return self


class BenchmarkArtifact(BenchmarkModel):
    schema_version: BenchmarkSchemaVersion = BENCHMARK_SCHEMA_VERSION
    definition: BenchmarkDefinition
    leakage_audit: BenchmarkLeakageAudit | None = None
    admission: BenchmarkAdmission

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @model_validator(mode="after")
    def admission_cannot_outrun_evidence(self) -> BenchmarkArtifact:
        definition = self.definition
        admission = self.admission
        if admission.definition_fingerprint != definition.fingerprint:
            raise ValueError("admission must bind the exact benchmark definition")
        evidence = {item.binding_id: item for item in definition.evidence_bindings}
        resolutions = {item.evidence_binding_id: item for item in admission.evidence_resolutions}
        if set(resolutions) != set(evidence):
            raise ValueError("admission must re-resolve every exact evidence binding")
        if any(
            resolutions[binding_id].assessment_reference != binding.assessment_reference
            for binding_id, binding in evidence.items()
        ):
            raise ValueError("admission resolution must preserve exact assessment references")

        baselines = {item.baseline_id: item for item in definition.baselines}
        metrics = {item.metric_id: item for item in definition.metrics}
        partition_ids = (
            {partition.partition_id for partition in definition.split_plan.partitions}
            if definition.split_plan is not None
            else set()
        )
        case_memberships = (
            {
                item.partition_id: item.evaluation_unit_ids
                for item in definition.evaluation_case_set.partition_memberships
            }
            if definition.evaluation_case_set is not None
            else {}
        )

        def validate_metric_result(result: MetricResult, *, owner: str) -> None:
            metric = metrics.get(result.metric.metric_id)
            if (
                metric is None
                or result.metric.metric_version != metric.metric_version
                or result.metric.metric_fingerprint != metric.fingerprint
            ):
                raise ValueError(f"{owner} metric results must bind exact metric definitions")
            if result.partition_id not in partition_ids:
                raise ValueError(f"{owner} metric result references an unknown partition")
            if result.partition_id not in metric.evaluation_partition_ids:
                raise ValueError(
                    f"{owner} metric result references an unfrozen evaluation partition"
                )
            expected_cases = case_memberships.get(result.partition_id)
            if expected_cases is None or result.evaluated_case_ids != expected_cases:
                raise ValueError(
                    f"{owner} metric result must cover the exact authoritative case membership"
                )
            if result.evaluated_evaluation_units != expected_cases.id_count:
                raise ValueError(f"{owner} metric result evaluation-unit count must be exact")
            if result.evaluated_evaluation_units < metric.minimum_evaluation_units:
                raise ValueError("metric result has too few declared evaluation units")
            if result.lower_confidence_bound is None or result.upper_confidence_bound is None:
                raise ValueError("metric results require both uncertainty bounds")

        runs = {item.baseline.baseline_id: item for item in admission.baseline_runs}
        if set(runs) != set(baselines):
            raise ValueError("admission must report every declared baseline")
        for baseline_id, baseline in baselines.items():
            run = runs[baseline_id]
            if (
                run.baseline.baseline_version != baseline.baseline_version
                or run.baseline.baseline_fingerprint != baseline.fingerprint
                or run.applicability_rule_fingerprint != baseline.applicability.fingerprint
            ):
                raise ValueError("baseline run must bind its exact definition and applicability")
            applies = baseline.applicability.applies_to(definition.query.state_query)
            if (run.status is BaselineRunStatus.NOT_APPLICABLE) is applies:
                raise ValueError("baseline N/A status must be derived from the frozen StateQuery")
            for result in run.metric_results:
                validate_metric_result(result, owner="baseline")
        for result in admission.metric_results:
            validate_metric_result(result, owner="benchmark")
        for comparison in admission.paired_baseline_comparisons:
            paired_metric = metrics.get(comparison.metric.metric_id)
            paired_baseline = baselines.get(comparison.baseline.baseline_id)
            if (
                paired_metric is None
                or comparison.metric.metric_version != paired_metric.metric_version
                or comparison.metric.metric_fingerprint != paired_metric.fingerprint
            ):
                raise ValueError("paired comparison must bind an exact metric definition")
            if (
                paired_baseline is None
                or comparison.baseline.baseline_version != paired_baseline.baseline_version
                or comparison.baseline.baseline_fingerprint != paired_baseline.fingerprint
            ):
                raise ValueError("paired comparison must bind an exact baseline definition")
            expected_cases = case_memberships.get(comparison.partition_id)
            if expected_cases is None or comparison.evaluated_case_ids != expected_cases:
                raise ValueError("paired comparison must cover exact authoritative case membership")
            expected_dependence = {
                item.dependence_id for item in paired_metric.uncertainty.dependence_units
            }
            if set(comparison.dependence_ids) != expected_dependence:
                raise ValueError("paired comparison must bind every exact metric dependence block")
            candidate_result = next(
                (
                    result
                    for result in admission.metric_results
                    if result.metric.metric_id == comparison.metric.metric_id
                    and result.partition_id == comparison.partition_id
                ),
                None,
            )
            baseline_run = runs.get(comparison.baseline.baseline_id)
            baseline_result = (
                next(
                    (
                        result
                        for result in baseline_run.metric_results
                        if result.metric.metric_id == comparison.metric.metric_id
                        and result.partition_id == comparison.partition_id
                    ),
                    None,
                )
                if baseline_run is not None
                else None
            )
            if candidate_result is None or baseline_result is None:
                raise ValueError(
                    "paired comparison requires exact candidate and baseline point results"
                )
            expected_effect = candidate_result.value - baseline_result.value
            if comparison.effect_scale is BaselineMarginMode.RELATIVE_FRACTION:
                if baseline_result.value == 0:
                    raise ValueError(
                        "relative paired effect is undefined for a zero baseline value"
                    )
                expected_effect /= abs(baseline_result.value)
            if not math.isclose(
                comparison.point_effect,
                expected_effect,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "paired point effect must equal the exact candidate-minus-baseline result"
                )

        if self.leakage_audit is None:
            if admission.leakage_audit_fingerprint is not None:
                raise ValueError("admission cannot reference a missing leakage audit")
        elif (
            definition.split_plan is None
            or self.leakage_audit.split_plan_fingerprint != definition.split_plan.fingerprint
            or admission.leakage_audit_fingerprint != self.leakage_audit.fingerprint
        ):
            raise ValueError("leakage audit must bind the exact split and admission")

        if admission.status is BenchmarkAdmissionStatus.TECHNICAL_ONLY:
            if definition.intent is not BenchmarkIntent.TECHNICAL_FIXTURE:
                raise ValueError("technical-only status requires explicit technical-fixture intent")
        elif admission.status is BenchmarkAdmissionStatus.COMPONENT_BENCHMARK:
            if definition.intent is not BenchmarkIntent.COMPONENT_BENCHMARK:
                raise ValueError("component status requires explicit component-benchmark intent")
        elif admission.status is BenchmarkAdmissionStatus.ADMITTED and definition.intent in {
            BenchmarkIntent.TECHNICAL_FIXTURE,
            BenchmarkIntent.COMPONENT_BENCHMARK,
        }:
            raise ValueError("technical or component intent cannot claim full scientific admission")

        if admission.status is BenchmarkAdmissionStatus.ADMITTED:
            self._validate_admitted_structure()
        return self

    def _validate_admitted_structure(self) -> None:
        definition = self.definition
        admission = self.admission
        if definition.design_status is not BenchmarkLifecycle.FROZEN:
            raise ValueError("only frozen definitions may be scientifically admitted")
        if definition.intent is not BenchmarkIntent.SCIENTIFIC:
            raise ValueError("scientific admission requires scientific benchmark intent")
        if any(
            binding.representability_proof_fingerprint is None
            for binding in definition.evidence_bindings
        ):
            raise ValueError("scientific admission requires exact representability proofs")
        assessment_identities = tuple(
            binding.assessment_identity for binding in definition.evidence_bindings
        )
        if not any(
            isinstance(identity, LossAssessmentIdentity) for identity in assessment_identities
        ) or not any(
            isinstance(identity, MetricAssessmentIdentity) for identity in assessment_identities
        ):
            raise ValueError("scientific admission requires exact eligible loss and metric refs")
        plan = definition.split_plan
        if plan is None or not definition.metrics or not definition.baselines:
            raise ValueError("admitted benchmarks require split, metric, and baseline definitions")
        expected_metric_results = {
            (metric.metric_id, partition_id)
            for metric in definition.metrics
            for partition_id in metric.evaluation_partition_ids
        }
        actual_metric_results = {
            (result.metric.metric_id, result.partition_id) for result in admission.metric_results
        }
        if actual_metric_results != expected_metric_results:
            raise ValueError(
                "scientific admission requires exactly one result for every declared "
                "metric and evaluation partition"
            )
        if (
            any(
                not isinstance(metric.implementation_binding, ExecutableImplementationBinding)
                for metric in definition.metrics
            )
            or any(
                not isinstance(baseline.implementation_binding, ExecutableImplementationBinding)
                for baseline in definition.baselines
            )
            or any(
                not isinstance(metric.uncertainty.method, ExecutableImplementationBinding)
                for metric in definition.metrics
            )
        ):
            raise ValueError(
                "scientific admission requires executable metric and baseline implementations"
            )
        if not definition.acceptance_rules or definition.acceptance_policy is None:
            raise ValueError("admitted benchmarks require a prespecified acceptance policy")
        roles = {partition.role for partition in plan.partitions}
        required_roles = {
            BenchmarkPartitionRole.TRAIN,
            BenchmarkPartitionRole.CALIBRATION,
            BenchmarkPartitionRole.MODEL_SELECTION_VALIDATION,
            BenchmarkPartitionRole.UNTOUCHED_TEST,
        }
        if not required_roles <= roles:
            raise ValueError(
                "admitted benchmarks require train, calibration, validation, and untouched test"
            )
        if any(partition.materialized_membership is None for partition in plan.partitions):
            raise ValueError(
                "generator identity is not partition membership; materialize every split"
            )
        if any(
            metric.aggregation is MetricAggregation.POOLED_RECORD
            or metric.weighting.scheme is MetricWeightingScheme.RECORD_COUNT_WEIGHTED
            for metric in definition.metrics
        ):
            raise ValueError("scientific metrics must not weight by record or cell count")
        if self.leakage_audit is None:
            raise ValueError("admitted benchmarks require a complete leakage audit")
        self._validate_leakage_audit(plan, self.leakage_audit)
        applicable_runs = []
        for baseline, run in zip(definition.baselines, admission.baseline_runs, strict=True):
            if baseline.applicability.applies_to(definition.query.state_query):
                applicable_runs.append(run)
                if run.status is not BaselineRunStatus.PASSED:
                    raise ValueError(
                        "failed, crashed, or unrun applicable baselines block admission"
                    )
        if not applicable_runs:
            raise ValueError("admitted benchmarks require at least one applicable passing baseline")
        if not _acceptance_policy_passes(definition, admission):
            raise ValueError("benchmark metric results do not pass the frozen acceptance policy")

    @staticmethod
    def _validate_leakage_audit(
        plan: BenchmarkSplitPlan,
        audit: BenchmarkLeakageAudit,
    ) -> None:
        partition_ids = tuple(partition.partition_id for partition in plan.partitions)
        if audit.evaluated_partition_ids != partition_ids:
            raise ValueError("leakage audit must cover every partition exactly")
        if any(check.status is not AuditCheckStatus.PASSED for check in audit.checks):
            raise ValueError("failed or unassessed leakage checks block admission")
        checks = audit.checks
        global_required = {
            LeakageCheckKind.SOURCE_DUPLICATE_DISJOINT,
            LeakageCheckKind.PREPROCESSING_FIT_ISOLATED,
            LeakageCheckKind.TARGET_DERIVATION_ISOLATED,
            LeakageCheckKind.TEMPORAL_CUTOFF_RESPECTED,
        }
        global_actual = {
            check.kind for check in checks if check.physical_dataset_binding_id is None
        }
        if not global_required <= global_actual:
            raise ValueError("leakage audit is missing a required global check")
        for kind in global_required:
            matches = tuple(
                check
                for check in checks
                if check.kind is kind and check.physical_dataset_binding_id is None
            )
            if len(matches) != 1 or matches[0].partition_ids != partition_ids:
                raise ValueError("leakage audit requires one complete check per global gate")
        source_check = next(
            check
            for check in checks
            if check.kind is LeakageCheckKind.SOURCE_DUPLICATE_DISJOINT
            and check.physical_dataset_binding_id is None
        )
        all_pair_count = len(partition_ids) * (len(partition_ids) - 1) // 2
        if source_check.comparisons_expected != all_pair_count:
            raise ValueError("source-duplicate audit must cover every partition pair")
        for closure in plan.protected_group_closures:
            ids = tuple(
                partition.partition_id
                for partition in plan.partitions
                if partition.physical_dataset_binding_id == closure.physical_dataset_binding_id
            )
            record_checks = tuple(
                check
                for check in checks
                if check.kind is LeakageCheckKind.RECORD_MEMBERSHIP_DISJOINT
                and check.physical_dataset_binding_id == closure.physical_dataset_binding_id
            )
            if len(record_checks) != 1 or record_checks[0].partition_ids != ids:
                raise ValueError("audit requires one complete record-overlap check per evidence")
            for group in closure.protected_groups:
                group_checks = tuple(
                    check
                    for check in checks
                    if check.kind is LeakageCheckKind.PROTECTED_GROUP_DISJOINT
                    and check.physical_dataset_binding_id == closure.physical_dataset_binding_id
                    and check.protected_unit == group.unit
                )
                if len(group_checks) != 1 or group_checks[0].partition_ids != ids:
                    raise ValueError("audit requires one complete check per protected group")
            expected_comparisons = len(ids) * (len(ids) - 1) // 2
            scoped_overlap_checks = tuple(
                check
                for check in checks
                if check.physical_dataset_binding_id == closure.physical_dataset_binding_id
                and check.kind
                in {
                    LeakageCheckKind.RECORD_MEMBERSHIP_DISJOINT,
                    LeakageCheckKind.PROTECTED_GROUP_DISJOINT,
                }
            )
            if any(
                check.comparisons_expected != expected_comparisons
                for check in scoped_overlap_checks
            ):
                raise ValueError("overlap audit must cover every within-evidence partition pair")


def _metric_result_estimate(result: MetricResult, estimate: ThresholdEstimate) -> float:
    if estimate is ThresholdEstimate.POINT:
        return result.value
    if estimate is ThresholdEstimate.LOWER_CONFIDENCE_BOUND:
        if result.lower_confidence_bound is None:
            raise ValueError("acceptance rule requires a lower confidence bound")
        return result.lower_confidence_bound
    if result.upper_confidence_bound is None:
        raise ValueError("acceptance rule requires an upper confidence bound")
    return result.upper_confidence_bound


def _threshold_comparison_passes(
    value: float,
    comparison: ThresholdComparison,
    threshold: float,
) -> bool:
    if comparison is ThresholdComparison.LESS_THAN:
        return value < threshold
    if comparison is ThresholdComparison.LESS_THAN_OR_EQUAL:
        return value <= threshold
    if comparison is ThresholdComparison.GREATER_THAN:
        return value > threshold
    return value >= threshold


def _acceptance_rule_passes(
    rule: BenchmarkAcceptanceRule,
    definition: BenchmarkDefinition,
    admission: BenchmarkAdmission,
) -> bool:
    candidate = next(
        (
            result
            for result in admission.metric_results
            if result.metric.metric_id == rule.metric.metric_id
            and result.partition_id == rule.partition_id
        ),
        None,
    )
    if candidate is None:
        return False
    if rule.absolute_threshold is not None:
        candidate_value = _metric_result_estimate(candidate, rule.estimate)
        return _threshold_comparison_passes(
            candidate_value,
            rule.comparison,
            rule.absolute_threshold,
        )

    assert rule.baseline_comparator is not None
    assert rule.baseline_margin is not None
    assert rule.baseline_margin_mode is not None
    assert rule.baseline_requirement is not None
    assert rule.confidence_level is not None
    baseline_refs = (
        (rule.baseline_comparator.baseline,)
        if isinstance(rule.baseline_comparator, ExactBaselineComparator)
        else rule.baseline_comparator.baselines
    )
    definitions = {item.baseline_id: item for item in definition.baselines}
    applicable_results: list[tuple[BaselineDefinitionReference, float]] = []
    for baseline_ref in baseline_refs:
        baseline_definition = definitions[baseline_ref.baseline_id]
        if not baseline_definition.applicability.applies_to(definition.query.state_query):
            continue
        run = next(
            (
                item
                for item in admission.baseline_runs
                if item.baseline.baseline_id == baseline_ref.baseline_id
            ),
            None,
        )
        if run is None or run.status is not BaselineRunStatus.PASSED:
            return False
        baseline_result = next(
            (
                result
                for result in run.metric_results
                if result.metric.metric_id == rule.metric.metric_id
                and result.partition_id == rule.partition_id
            ),
            None,
        )
        if baseline_result is None:
            return False
        applicable_results.append((baseline_ref, baseline_result.value))
    if not applicable_results:
        return False
    metric = next(item for item in definition.metrics if item.metric_id == rule.metric.metric_id)
    if isinstance(rule.baseline_comparator, ExactBaselineComparator):
        selected_baseline = applicable_results[0][0]
    elif metric.direction is MetricDirection.MINIMIZE:
        selected_baseline = min(applicable_results, key=lambda item: item[1])[0]
    else:
        selected_baseline = max(applicable_results, key=lambda item: item[1])[0]
    paired = next(
        (
            item
            for item in admission.paired_baseline_comparisons
            if item.metric.metric_id == rule.metric.metric_id
            and item.partition_id == rule.partition_id
            and item.baseline.baseline_id == selected_baseline.baseline_id
        ),
        None,
    )
    expected_bound = (
        PairedConfidenceBoundKind.UPPER
        if metric.direction is MetricDirection.MINIMIZE
        else PairedConfidenceBoundKind.LOWER
    )
    if (
        paired is None
        or paired.effect_scale is not rule.baseline_margin_mode
        or paired.bound_kind is not expected_bound
        or paired.confidence_level != rule.confidence_level
    ):
        return False
    threshold = (
        rule.baseline_margin
        if rule.baseline_requirement is BaselineRequirement.NONINFERIOR
        else -rule.baseline_margin
    )
    if metric.direction is MetricDirection.MAXIMIZE:
        threshold = -threshold
    return _threshold_comparison_passes(
        paired.one_sided_confidence_bound,
        rule.comparison,
        threshold,
    )


def _acceptance_policy_passes(
    definition: BenchmarkDefinition,
    admission: BenchmarkAdmission,
) -> bool:
    policy = definition.acceptance_policy
    if policy is None:
        return False
    rule_results = {
        rule.rule_id: _acceptance_rule_passes(rule, definition, admission)
        for rule in definition.acceptance_rules
    }
    groups = {group.group_id: group for group in policy.groups}

    def evaluate(group_id: str) -> bool:
        group = groups[group_id]
        members = [rule_results[rule_id] for rule_id in group.rule_ids]
        members.extend(evaluate(child_id) for child_id in group.child_group_ids)
        if group.operator is AcceptanceGroupOperator.ALL:
            return all(members)
        return any(members)

    return evaluate(policy.root_group_id)


class BenchmarkVerification(BenchmarkModel):
    artifact_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    declared_status: BenchmarkAdmissionStatus
    evidence_resolutions_verified: bool
    assessment_and_permission_gates_passed: bool
    performance_gates_passed: bool
    admission_ready: bool
    technical_fixture_eligible: bool
    verified: bool
    blockers: tuple[str, ...] = ()

    @field_validator("artifact_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()


def _resolution_fingerprint(resolution: DatasetAssessmentResolution) -> str:
    return canonical_fingerprint(resolution.model_dump(mode="json"))


def verify_benchmark_artifact(
    artifact: BenchmarkArtifact,
    manifests: Mapping[str, DatasetManifest],
) -> BenchmarkVerification:
    """Re-resolve exact manifests and fail if a declared admission outruns science or law."""

    definition = artifact.definition
    expected_ids = tuple(binding.binding_id for binding in definition.evidence_bindings)
    if set(manifests) != set(expected_ids):
        raise ValueError("benchmark verification requires every and only bound dataset manifests")
    resolution_bindings = {
        item.evidence_binding_id: item for item in artifact.admission.evidence_resolutions
    }
    blockers: list[str] = []
    assessment_permission_pass = True
    resolved_claims: dict[str, ClaimAssessment] = {}
    for binding in definition.evidence_bindings:
        manifest = manifests[binding.binding_id]
        if (
            manifest.dataset_id != binding.dataset_id
            or manifest.version != binding.dataset_version
            or manifest.fingerprint != binding.manifest_fingerprint
        ):
            raise ValueError("benchmark evidence binding does not match supplied manifest")
        canonical_manifest_bytes = _canonical_model_bytes(manifest)
        if (
            binding.manifest_artifact.sha256 != manifest.fingerprint
            or binding.manifest_artifact.byte_count != len(canonical_manifest_bytes)
            or binding.manifest_artifact.media_type != "application/json"
        ):
            raise ValueError("manifest artifact must contain the exact canonical manifest bytes")
        resolution = manifest.resolve_assessment(
            binding.assessment_reference,
            use_case=DataUseCase.BENCHMARK_EVALUATION,
        )
        if resolution.assessment_kind is not binding.assessment_kind:
            raise ValueError("benchmark evidence assessment kind does not match the manifest")
        assessment = _assessment_for_binding(manifest, binding)
        _verify_assessment_identity(binding, assessment)
        expected_split_unit = (
            assessment.required_split_unit
            if isinstance(assessment, ObjectiveEligibilityAssessment)
            else manifest.experimental_design.default_split_unit
        )
        if binding.required_split_unit.level is not expected_split_unit:
            raise ValueError("benchmark split requirement does not match its exact assessment")
        frozen = resolution_bindings[binding.binding_id]
        if frozen.resolution_fingerprint != _resolution_fingerprint(resolution):
            raise ValueError("benchmark evidence resolution has changed since admission review")
        passed = (
            resolution.scientific_status is EligibilityStatus.ELIGIBLE
            and resolution.effective_permission is not None
            and resolution.effective_permission.status is PermissionStatus.PERMITTED
            and resolution.use_allowed_without_additional_review
        )
        if not passed:
            assessment_permission_pass = False
            blockers.append(
                f"{binding.binding_id}: scientific/legal benchmark resolution did not pass"
            )
        _verify_split_binding(definition, binding, manifest)
        _verify_scope_binding(definition, binding, manifest)
        for claim in _claim_assessments_for_binding(manifest, binding):
            resolved_claims[claim.assessment_id] = claim

    declared = artifact.admission.status
    if declared is BenchmarkAdmissionStatus.ADMITTED and not assessment_permission_pass:
        raise ValueError("ADMITTED requires every scientific and legal resolution to pass")
    if declared is BenchmarkAdmissionStatus.ADMITTED:
        _verify_causal_status(definition, tuple(resolved_claims.values()))
    if declared is BenchmarkAdmissionStatus.TECHNICAL_ONLY and (
        definition.intent is not BenchmarkIntent.TECHNICAL_FIXTURE
    ):
        raise ValueError("scientific intent cannot be silently downgraded to technical-only")

    leakage_pass = False
    if definition.split_plan is not None and artifact.leakage_audit is not None:
        try:
            BenchmarkArtifact._validate_leakage_audit(
                definition.split_plan,
                artifact.leakage_audit,
            )
        except ValueError:
            pass
        else:
            leakage_pass = True
    applicable_runs = tuple(
        run
        for baseline, run in zip(
            definition.baselines,
            artifact.admission.baseline_runs,
            strict=True,
        )
        if baseline.applicability.applies_to(definition.query.state_query)
    )
    baselines_pass = bool(applicable_runs) and all(
        run.status is BaselineRunStatus.PASSED for run in applicable_runs
    )
    implementations_ready = (
        bool(definition.metrics and definition.baselines)
        and all(
            isinstance(metric.implementation_binding, ExecutableImplementationBinding)
            for metric in definition.metrics
        )
        and all(
            isinstance(baseline.implementation_binding, ExecutableImplementationBinding)
            for baseline in definition.baselines
        )
        and all(
            isinstance(metric.uncertainty.method, ExecutableImplementationBinding)
            for metric in definition.metrics
        )
    )
    acceptance_pass = bool(
        definition.acceptance_rules
        and definition.acceptance_policy is not None
        and artifact.admission.metric_results
        and _acceptance_policy_passes(definition, artifact.admission)
    )
    performance_pass = leakage_pass and baselines_pass and implementations_ready and acceptance_pass
    admission_ready = (
        assessment_permission_pass
        and performance_pass
        and definition.design_status is BenchmarkLifecycle.FROZEN
        and definition.intent is BenchmarkIntent.SCIENTIFIC
    )
    if not performance_pass:
        blockers.append("benchmark performance gates are incomplete or did not pass")
    if declared is BenchmarkAdmissionStatus.ADMITTED and not admission_ready:
        raise ValueError("ADMITTED requires assessment, permission, and performance gates to pass")
    technical_eligible = definition.intent is BenchmarkIntent.TECHNICAL_FIXTURE
    return BenchmarkVerification(
        artifact_fingerprint=artifact.fingerprint,
        declared_status=declared,
        evidence_resolutions_verified=True,
        assessment_and_permission_gates_passed=assessment_permission_pass,
        performance_gates_passed=performance_pass,
        admission_ready=admission_ready,
        technical_fixture_eligible=technical_eligible,
        verified=True,
        blockers=tuple(sorted(blockers)),
    )


def _verify_split_binding(
    definition: BenchmarkDefinition,
    binding: BenchmarkEvidenceBinding,
    manifest: DatasetManifest,
) -> None:
    plan = definition.split_plan
    if plan is None:
        return
    used_datasets = {partition.physical_dataset_binding_id for partition in plan.partitions}
    if binding.physical_dataset_binding_id not in used_datasets:
        return
    universe = next(
        item
        for item in plan.universes
        if item.physical_dataset_binding_id == binding.physical_dataset_binding_id
    )
    closure = next(
        item
        for item in plan.protected_group_closures
        if item.physical_dataset_binding_id == binding.physical_dataset_binding_id
    )
    if universe.slice_fingerprint != manifest.slice_spec.fingerprint:
        raise ValueError("benchmark universe does not bind the exact dataset slice")
    if universe.record_ids.id_count != manifest.slice_spec.selected_record_count:
        raise ValueError("benchmark universe record count does not match the dataset slice")
    units = {unit.level: unit for unit in manifest.experimental_design.units}
    required = units.get(binding.required_split_unit.level)
    if required is None or required.resolved_identity != binding.required_split_unit.identity:
        raise ValueError("benchmark required split unit does not match the dataset manifest")
    record_matches = tuple(
        unit
        for unit in manifest.experimental_design.units
        if unit.level is universe.record_unit.level
        and unit.resolved_identity == universe.record_unit.identity
    )
    if len(record_matches) != 1 or record_matches[0].level is not universe.record_unit.level:
        raise ValueError("benchmark record unit does not resolve uniquely in the manifest")
    chain: list[ExperimentalUnitBinding] = []
    current = record_matches[0]
    while True:
        chain.append(
            ExperimentalUnitBinding(level=current.level, identity=current.resolved_identity)
        )
        if current.parent_level is None:
            break
        current = units[current.parent_level]
    expected_ancestry = tuple(reversed(chain))
    if closure.unit_ancestry != expected_ancestry:
        raise ValueError("protected-group closure must reproduce exact manifest unit ancestry")
    protected = {group.unit.level: group for group in closure.protected_groups}
    required_reasons: dict[ExperimentalUnitLevel, set[ProtectedGroupReason]] = {}

    def require(level: ExperimentalUnitLevel | None, reason: ProtectedGroupReason) -> None:
        if level is not None:
            required_reasons.setdefault(level, set()).add(reason)

    design = manifest.experimental_design
    require(design.default_split_unit, ProtectedGroupReason.DEFAULT_SPLIT)
    require(design.sampling.subject_unit, ProtectedGroupReason.SAMPLING_SUBJECT)
    require(design.biological_replicate_unit, ProtectedGroupReason.BIOLOGICAL_REPLICATE)
    require(design.randomization_unit, ProtectedGroupReason.RANDOMIZATION)
    require(binding.required_split_unit.level, ProtectedGroupReason.OBJECTIVE_REQUIRED_SPLIT)
    require(closure.assignment_unit.level, ProtectedGroupReason.SPLIT_ASSIGNMENT)
    for level, reasons in required_reasons.items():
        group = protected.get(level)
        if group is None or not reasons <= set(group.reasons):
            raise ValueError("protected-group closure omits a manifest-derived sharing boundary")


def _assessment_for_binding(
    manifest: DatasetManifest,
    binding: BenchmarkEvidenceBinding,
) -> ClaimAssessment | ObjectiveEligibilityAssessment:
    assessment_id = binding.assessment_reference.assessment_id
    if binding.assessment_kind is AssessmentKind.CLAIM:
        return next(
            assessment
            for assessment in manifest.claim_assessments
            if assessment.assessment_id == assessment_id
        )
    if binding.assessment_kind is AssessmentKind.LOSS:
        return next(
            assessment
            for assessment in manifest.loss_assessments
            if assessment.assessment_id == assessment_id
        )
    return next(
        assessment
        for assessment in manifest.metric_assessments
        if assessment.assessment_id == assessment_id
    )


def _verify_assessment_identity(
    binding: BenchmarkEvidenceBinding,
    assessment: ClaimAssessment | ObjectiveEligibilityAssessment,
) -> None:
    identity = binding.assessment_identity
    if isinstance(identity, ClaimAssessmentIdentity):
        if not isinstance(assessment, ClaimAssessment) or identity.claim is not assessment.claim:
            raise ValueError("claim assessment identity does not match the supplied manifest")
        return
    if isinstance(identity, LossAssessmentIdentity):
        if (
            not isinstance(assessment, LossEligibilityAssessment)
            or identity.loss_kind is not assessment.loss_kind
        ):
            raise ValueError("loss assessment identity does not match the supplied manifest")
        return
    if not isinstance(assessment, MetricEligibilityAssessment) or (
        identity.metric_id != assessment.metric_id
        or identity.metric_family is not assessment.metric_family
        or identity.partition_purpose is not assessment.partition_purpose
    ):
        raise ValueError("metric assessment identity does not match the supplied manifest")


def _claim_assessments_for_binding(
    manifest: DatasetManifest,
    binding: BenchmarkEvidenceBinding,
) -> tuple[ClaimAssessment, ...]:
    assessment = _assessment_for_binding(manifest, binding)
    if isinstance(assessment, ClaimAssessment):
        return (assessment,)
    claims_by_id = {claim.assessment_id: claim for claim in manifest.claim_assessments}
    return tuple(
        claims_by_id[reference.assessment_id]
        for reference in assessment.supporting_claim_assessments
    )


def _verify_causal_status(
    definition: BenchmarkDefinition,
    claims: tuple[ClaimAssessment, ...],
) -> None:
    scope = definition.scope
    for status in (
        scope.reference_estimand_causal_status,
        scope.forecast_causal_status,
    ):
        if status is CausalStatus.IDENTIFIED_POPULATION_EFFECT and not any(
            claim.claim is ScientificClaim.INTERVENTION_EFFECT
            and claim.identification_basis is IdentificationBasis.RANDOMIZED_WITHIN_STUDY
            for claim in claims
        ):
            raise ValueError(
                "identified-effect admission requires an exact randomized intervention claim"
            )
        if status is CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS and not any(
            claim.claim is ScientificClaim.COUNTERFACTUAL_GENERALIZATION
            and claim.identification_basis is IdentificationBasis.TRANSPORTED_UNDER_ASSUMPTIONS
            for claim in claims
        ):
            raise ValueError(
                "transported admission requires an exact transported counterfactual claim"
            )
        if status is CausalStatus.MECHANISTIC_EXTRAPOLATION:
            raise ValueError("dataset assessment contracts cannot admit mechanistic extrapolation")


def _verify_scope_binding(
    definition: BenchmarkDefinition,
    binding: BenchmarkEvidenceBinding,
    manifest: DatasetManifest,
) -> None:
    assessment = _assessment_for_binding(manifest, binding)
    query = definition.query.state_query
    scope_binding = binding.scope_binding
    if scope_binding.assessment_scope_fingerprint != assessment.scope.fingerprint:
        raise ValueError("evidence mapping does not bind the exact assessment scope")
    expected_subject = {
        SubjectKind.INDIVIDUAL_CELL: "individual_cell",
        SubjectKind.CLONE_LINEAGE: "clone",
        SubjectKind.POPULATION: "population",
        SubjectKind.SPATIAL_NICHE: "spatial_region",
    }[query.subject.kind]
    if assessment.scope.subject_kind.value != expected_subject:
        raise ValueError("benchmark query subject does not match the evidence assessment")
    if assessment.scope.system_boundary is not query.system_boundary:
        raise ValueError("benchmark query boundary does not match the evidence assessment")
    if query.subject.biological_system.key not in {
        term.key for term in assessment.scope.biological_systems
    }:
        raise ValueError("benchmark biological system is outside the evidence assessment")
    query_targets = {item.term.key: item for item in query.target_outputs}
    units_by_level = {unit.level: unit for unit in manifest.experimental_design.units}
    unit_bindings = {
        ExperimentalUnitBinding(level=unit.level, identity=unit.resolved_identity): unit
        for unit in manifest.experimental_design.units
    }
    modalities_by_key = {item.modality.key: item for item in manifest.capabilities.modalities}
    readouts_by_id = {item.readout_id: item for item in manifest.capabilities.functional.outputs}
    mapped_modalities: set[str] = set()
    mapped_readouts: set[str] = set()
    for mapping in scope_binding.target_mappings:
        output = query_targets[mapping.target_output_key]
        if mapping.aggregation_unit not in unit_bindings:
            raise ValueError("target aggregation mapping references an undeclared dataset unit")
        mapped_modalities.update(mapping.assessment_modalities)
        mapped_readouts.update(mapping.assessment_functional_readout_ids)
        for modality_key in mapping.assessment_modalities:
            modality = modalities_by_key.get(modality_key)
            if modality is None:
                raise ValueError("target mapping references an undeclared dataset modality")
            if modality.alignment_unit is not None:
                aligned_units = [units_by_level[modality.alignment_unit]]
            elif modality.alignment_identity is not None:
                aligned_units = [
                    unit
                    for unit in manifest.experimental_design.units
                    if unit.resolved_identity == modality.alignment_identity
                ]
            elif modality.alignment_key_field is not None:
                aligned_units = [
                    unit
                    for unit in manifest.experimental_design.units
                    if unit.resolved_identity.source_fields == (modality.alignment_key_field,)
                ]
            else:
                aligned_units = []
            if modality.subject_alignment.value != "unpaired":
                if len(aligned_units) != 1:
                    raise ValueError("modality alignment does not resolve to one dataset unit")
                current = aligned_units[0]
                ancestors: set[ExperimentalUnitLevel] = {current.level}
                while current.parent_level is not None:
                    current = units_by_level[current.parent_level]
                    ancestors.add(current.level)
                if mapping.aggregation_unit.level not in ancestors:
                    raise ValueError(
                        "target aggregation must be an ancestor of its measured modality"
                    )
        for readout_id in mapping.assessment_functional_readout_ids:
            readout = readouts_by_id.get(readout_id)
            if readout is None:
                raise ValueError("target mapping references an undeclared functional readout")
            if (
                readout.output.key != output.term.key
                or readout.units != output.units
                or readout.aggregation_level is not mapping.aggregation_unit.level
            ):
                raise ValueError(
                    "functional readout does not match exact target term, units, and aggregation"
                )
    if mapped_modalities != {term.key for term in assessment.scope.modalities}:
        raise ValueError("target mappings must exactly cover assessment-scope modalities")
    if mapped_readouts != set(assessment.scope.functional_readout_ids):
        raise ValueError("target mappings must exactly cover assessment functional readouts")

    mapped_intervention_kinds = {
        item.assessment_intervention_kind_key for item in scope_binding.intervention_mappings
    }
    if mapped_intervention_kinds != {term.key for term in assessment.scope.intervention_kinds}:
        raise ValueError("intervention mappings must exactly cover assessment action kinds")
    mapped_environment_keys = {
        item.assessment_environment_variable_key for item in scope_binding.environment_mappings
    }
    if mapped_environment_keys != {term.key for term in assessment.scope.environment_variables}:
        raise ValueError("environment mappings must exactly cover assessment variables")

    actual_claims = tuple(
        sorted(
            (claim.claim for claim in _claim_assessments_for_binding(manifest, binding)),
            key=lambda claim: claim.value,
        )
    )
    if scope_binding.scientific_claims != actual_claims:
        raise ValueError("evidence claim classes do not match exact assessment claims")

    query_horizons = {item.name: item.duration_seconds for item in query.prediction_horizons}
    mapped_horizons = tuple(sorted(query_horizons[name] for name in scope_binding.horizon_names))
    if mapped_horizons != assessment.scope.horizons_seconds:
        raise ValueError("benchmark horizons do not exactly match the evidence assessment")
    if assessment.scope.horizon_windows:
        raise ValueError("v0.1 benchmark bindings do not silently coerce interval horizons")
    if (
        assessment.scope.inference_cutoff_seconds != definition.scope.inference_cutoff_seconds
        or assessment.scope.inference_cutoff_field != definition.scope.inference_cutoff_field
        or assessment.scope.inference_cutoff_window is not None
    ):
        raise ValueError("benchmark cutoff does not exactly match the evidence assessment")


__all__ = [
    "ADMISSION_SCHEMA_VERSION",
    "BASELINE_DEFINITION_SCHEMA_VERSION",
    "BENCHMARK_SCHEMA_VERSION",
    "EVALUATION_CASE_SET_SCHEMA_VERSION",
    "LEAKAGE_AUDIT_SCHEMA_VERSION",
    "METRIC_DEFINITION_SCHEMA_VERSION",
    "QUERY_BINDING_SCHEMA_VERSION",
    "SPLIT_PLAN_SCHEMA_VERSION",
    "AcceptanceGroupOperator",
    "AdmissionSchemaVersion",
    "AssessmentIdentity",
    "AuditCheckStatus",
    "BaselineApplicabilityRule",
    "BaselineComparator",
    "BaselineDefinitionReference",
    "BaselineDefinitionSchemaVersion",
    "BaselineMarginMode",
    "BaselineRequirement",
    "BaselineRun",
    "BaselineRunStatus",
    "BenchmarkAcceptanceGroup",
    "BenchmarkAcceptancePolicy",
    "BenchmarkAcceptanceRule",
    "BenchmarkAdmission",
    "BenchmarkAdmissionStatus",
    "BenchmarkArtifact",
    "BenchmarkBaselineDefinition",
    "BenchmarkDefinition",
    "BenchmarkEvaluationCase",
    "BenchmarkEvaluationCaseSet",
    "BenchmarkEvidenceBinding",
    "BenchmarkImplementationBinding",
    "BenchmarkIntent",
    "BenchmarkLeakageAudit",
    "BenchmarkLifecycle",
    "BenchmarkMetricDefinition",
    "BenchmarkParameter",
    "BenchmarkPartition",
    "BenchmarkPartitionRole",
    "BenchmarkSchemaVersion",
    "BenchmarkScope",
    "BenchmarkSplitPlan",
    "BenchmarkVerification",
    "BestApplicableBaselineComparator",
    "CanonicalIdMembership",
    "ClaimAssessmentIdentity",
    "ContentAddressedArtifact",
    "EvaluationCasePartitionBinding",
    "EvaluationCaseRole",
    "EvaluationCaseSetSchemaVersion",
    "EvaluationContextBinding",
    "EvaluationInterventionMultiplicity",
    "EvidenceEnvironmentMapping",
    "EvidenceInterventionMapping",
    "EvidenceResolutionBinding",
    "EvidenceScopeBinding",
    "EvidenceTargetMapping",
    "ExactBaselineComparator",
    "ExcludedPartitionMembership",
    "ExecutableImplementationBinding",
    "ExperimentalUnitBinding",
    "ExplicitPartitionMembership",
    "LeakageAuditCheck",
    "LeakageAuditSchemaVersion",
    "LeakageCheckKind",
    "LossAssessmentIdentity",
    "MetricAggregation",
    "MetricAssessmentIdentity",
    "MetricDefinitionReference",
    "MetricDefinitionSchemaVersion",
    "MetricDependenceKind",
    "MetricDependenceUnit",
    "MetricDirection",
    "MetricMissingnessPolicy",
    "MetricResamplingScheme",
    "MetricResult",
    "MetricUncertaintySpec",
    "MetricWeightingPolicy",
    "MetricWeightingScheme",
    "PairedBaselineComparisonResult",
    "PairedConfidenceBoundKind",
    "PartitionGenerationSpec",
    "PartitionMembershipSpec",
    "PartitionUniverse",
    "PredictionRepresentation",
    "ProtectedGroupBinding",
    "ProtectedGroupClosure",
    "ProtectedGroupMembership",
    "ProtectedGroupReason",
    "QueryBindingSchemaVersion",
    "QueryParameterAxis",
    "QueryParameterGrid",
    "QueryParameterValue",
    "SpecificationOnlyImplementationBinding",
    "SplitPlanSchemaVersion",
    "StateQueryBinding",
    "TargetRepresentation",
    "ThresholdComparison",
    "ThresholdEstimate",
    "VersionedImplementation",
    "verify_benchmark_artifact",
]
