"""Reviewed, content-bound proofs that a dataset can represent one scientific task.

Representability is a scientific evidence decision.  It neither evaluates data-use terms nor
authorizes a workflow; those decisions remain in the independent permission contracts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cellstate.domain.common import canonical_fingerprint
from cellstate.domain.query import SystemBoundary

from .manifests import (
    ClaimAssessment,
    ClaimAssessmentReference,
    DatasetAssessmentReference,
    DatasetManifest,
    DatasetSliceKind,
    EligibilityStatus,
    ExperimentalUnitLevel,
    ManifestModel,
    SamplingMode,
    SamplingSubjectKind,
    ScientificClaim,
    SubjectAlignment,
    SubjectLinkage,
)

RepresentabilityProofSchemaVersion = Literal["0.1-experimental"]
REPRESENTABILITY_PROOF_SCHEMA_VERSION: RepresentabilityProofSchemaVersion = "0.1-experimental"


def _canonical_id(value: str, *, name: str) -> str:
    if not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be nonempty and omit surrounding whitespace")
    return value


def _canonical_unique_sorted(values: tuple[str, ...], *, name: str) -> None:
    if any(not value.strip() or value != value.strip() for value in values):
        raise ValueError(f"{name} entries must be canonical nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")
    if tuple(sorted(values)) != values:
        raise ValueError(f"{name} must be sorted")


def canonical_selected_record_ids_sha256(record_ids: Sequence[str]) -> str:
    """Hash semantic membership as a sorted compact JSON UTF-8 string array.

    Input order and transport-file whitespace do not affect the result. Duplicate IDs and IDs
    with surrounding whitespace are rejected rather than silently normalized.
    """

    if isinstance(record_ids, str):
        raise TypeError("selected record IDs must be a sequence of strings, not one string")
    values = tuple(record_ids)
    if not values:
        raise ValueError("selected record IDs must not be empty")
    if any(not isinstance(value, str) for value in values):
        raise TypeError("selected record IDs must all be strings")
    if any(not value.strip() or value != value.strip() for value in values):
        raise ValueError("selected record IDs must be canonical nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError("selected record IDs must be unique")
    encoded = json.dumps(
        sorted(values),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RepresentabilityProofKind(StrEnum):
    DESTRUCTIVE_POPULATION = "destructive_population"
    INDIVIDUAL_CELL_FUNCTIONAL_RECORDER = "individual_cell_functional_recorder"


class RepresentabilityCriterion(StrEnum):
    EXACT_SLICE_BOUND = "exact_slice_bound"
    SOURCE_BYTES_BOUND = "source_bytes_bound"
    POPULATION_SUBJECT = "population_subject"
    DESTRUCTIVE_COLLECTION = "destructive_collection"
    POPULATION_LINKAGE_BOUNDARY = "population_linkage_boundary"
    INDIVIDUAL_CELL_SUBJECT = "individual_cell_subject"
    SAME_CELL_LINKAGE = "same_cell_linkage"
    VIABILITY_PRESERVING_COLLECTION = "viability_preserving_collection"
    BASELINE_BEFORE_TARGET = "baseline_before_target"
    FUTURE_SAME_CELL_FUNCTIONAL_READOUT = "future_same_cell_functional_readout"
    POPULATION_CAST_REJECTED = "population_cast_rejected"
    INDIVIDUAL_CAST_REJECTED = "individual_cast_rejected"
    CLONE_CAST_REJECTED = "clone_cast_rejected"
    CAUSAL_OVERCLAIM_REJECTED = "causal_overclaim_rejected"
    TRANSPORT_OVERCLAIM_REJECTED = "transport_overclaim_rejected"


class RepresentabilityCriterionStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class RepresentabilityEvidenceMethod(StrEnum):
    CHECKSUM_VERIFICATION = "checksum_verification"
    SELECTOR_EXECUTION = "selector_execution"
    SOURCE_FIELD = "source_field"
    PUBLICATION_METHOD = "publication_method"
    DIRECT_IMAGE_TRACKING = "direct_image_tracking"
    RECORDED_SAME_CELL_ID = "recorded_same_cell_id"
    MANIFEST_VALIDATION = "manifest_validation"


class RepresentabilitySourceBinding(ManifestModel):
    source_id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    @field_validator("source_id")
    @classmethod
    def source_id_is_canonical(cls, value: str) -> str:
        return _canonical_id(value, name="representability source ID")

    @field_validator("sha256")
    @classmethod
    def canonicalize_sha256(cls, value: str) -> str:
        return value.casefold()


class RepresentabilityEvidenceLocator(ManifestModel):
    source_id: str = Field(min_length=1)
    method: RepresentabilityEvidenceMethod
    locator: str = Field(min_length=1)

    @model_validator(mode="after")
    def locator_is_canonical(self) -> RepresentabilityEvidenceLocator:
        _canonical_id(self.source_id, name="representability evidence source ID")
        _canonical_id(self.locator, name="representability evidence locator")
        return self


class RepresentabilityCriterionTrace(ManifestModel):
    criterion: RepresentabilityCriterion
    status: RepresentabilityCriterionStatus
    evidence_locators: tuple[RepresentabilityEvidenceLocator, ...] = ()
    evidence_notes: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    @model_validator(mode="after")
    def review_is_auditable(self) -> RepresentabilityCriterionTrace:
        locator_keys = tuple(
            (locator.source_id, locator.method.value, locator.locator)
            for locator in self.evidence_locators
        )
        if len(locator_keys) != len(set(locator_keys)):
            raise ValueError("representability evidence locators must be unique")
        if tuple(sorted(locator_keys)) != locator_keys:
            raise ValueError("representability evidence locators must be sorted")
        _canonical_unique_sorted(self.evidence_notes, name="representability evidence notes")
        _canonical_unique_sorted(self.blockers, name="representability blockers")
        if self.status is RepresentabilityCriterionStatus.PASSED:
            if not self.evidence_locators or not self.evidence_notes:
                raise ValueError("passed representability criteria require locators and notes")
            if self.blockers:
                raise ValueError("passed representability criteria cannot retain blockers")
        elif not self.blockers:
            raise ValueError("failed representability criteria require blockers")
        return self


_POPULATION_CRITERIA = frozenset(
    {
        RepresentabilityCriterion.EXACT_SLICE_BOUND,
        RepresentabilityCriterion.SOURCE_BYTES_BOUND,
        RepresentabilityCriterion.POPULATION_SUBJECT,
        RepresentabilityCriterion.DESTRUCTIVE_COLLECTION,
        RepresentabilityCriterion.POPULATION_LINKAGE_BOUNDARY,
        RepresentabilityCriterion.INDIVIDUAL_CAST_REJECTED,
        RepresentabilityCriterion.CLONE_CAST_REJECTED,
        RepresentabilityCriterion.CAUSAL_OVERCLAIM_REJECTED,
        RepresentabilityCriterion.TRANSPORT_OVERCLAIM_REJECTED,
    }
)

_INDIVIDUAL_FUNCTIONAL_RECORDER_CRITERIA = frozenset(
    {
        RepresentabilityCriterion.EXACT_SLICE_BOUND,
        RepresentabilityCriterion.SOURCE_BYTES_BOUND,
        RepresentabilityCriterion.INDIVIDUAL_CELL_SUBJECT,
        RepresentabilityCriterion.SAME_CELL_LINKAGE,
        RepresentabilityCriterion.VIABILITY_PRESERVING_COLLECTION,
        RepresentabilityCriterion.BASELINE_BEFORE_TARGET,
        RepresentabilityCriterion.FUTURE_SAME_CELL_FUNCTIONAL_READOUT,
        RepresentabilityCriterion.POPULATION_CAST_REJECTED,
        RepresentabilityCriterion.CLONE_CAST_REJECTED,
        RepresentabilityCriterion.CAUSAL_OVERCLAIM_REJECTED,
        RepresentabilityCriterion.TRANSPORT_OVERCLAIM_REJECTED,
    }
)

_REQUIRED_CRITERIA = {
    RepresentabilityProofKind.DESTRUCTIVE_POPULATION: _POPULATION_CRITERIA,
    RepresentabilityProofKind.INDIVIDUAL_CELL_FUNCTIONAL_RECORDER: (
        _INDIVIDUAL_FUNCTIONAL_RECORDER_CRITERIA
    ),
}


class RepresentabilityProof(ManifestModel):
    schema_version: RepresentabilityProofSchemaVersion = REPRESENTABILITY_PROOF_SCHEMA_VERSION
    proof_id: str = Field(min_length=1)
    proof_kind: RepresentabilityProofKind
    assessment_reference: DatasetAssessmentReference
    slice_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    source_bindings: tuple[RepresentabilitySourceBinding, ...] = Field(min_length=1)
    criterion_traces: tuple[RepresentabilityCriterionTrace, ...] = Field(min_length=1)
    negative_claim_assessments: tuple[ClaimAssessmentReference, ...] = Field(min_length=1)
    reviewed_by: tuple[str, ...] = Field(min_length=1)
    reviewed_on: date

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.model_dump(mode="json"))

    @field_validator("proof_id")
    @classmethod
    def proof_id_is_canonical(cls, value: str) -> str:
        return _canonical_id(value, name="representability proof ID")

    @field_validator("slice_fingerprint")
    @classmethod
    def canonicalize_fingerprint(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def proof_is_complete_and_canonical(self) -> RepresentabilityProof:
        binding_ids = tuple(binding.source_id for binding in self.source_bindings)
        _canonical_unique_sorted(binding_ids, name="representability source bindings")
        criteria = tuple(trace.criterion for trace in self.criterion_traces)
        if len(criteria) != len(set(criteria)):
            raise ValueError("representability criteria must be unique")
        if tuple(sorted(criteria, key=lambda item: item.value)) != criteria:
            raise ValueError("representability criteria must be sorted")
        required = _REQUIRED_CRITERIA[self.proof_kind]
        if set(criteria) != required:
            missing = sorted(item.value for item in required - set(criteria))
            unexpected = sorted(item.value for item in set(criteria) - required)
            raise ValueError(
                "representability proof has an invalid criterion set: "
                f"missing={missing}, unexpected={unexpected}"
            )
        negative_ids = tuple(
            reference.assessment_id for reference in self.negative_claim_assessments
        )
        _canonical_unique_sorted(negative_ids, name="negative claim-assessment references")
        _canonical_unique_sorted(self.reviewed_by, name="representability reviewers")
        return self


class RepresentabilityResolution(ManifestModel):
    proof_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    manifest_fingerprint: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    proof_kind: RepresentabilityProofKind
    accepted: bool
    failed_criteria: tuple[RepresentabilityCriterion, ...] = ()
    structurally_failed_criteria: tuple[RepresentabilityCriterion, ...] = ()
    use_permission_evaluated: Literal[False] = False
    use_authorized: Literal[False] = False

    @field_validator("proof_fingerprint", "manifest_fingerprint")
    @classmethod
    def canonicalize_fingerprints(cls, value: str) -> str:
        return value.casefold()

    @model_validator(mode="after")
    def outcome_is_coherent(self) -> RepresentabilityResolution:
        for criteria, name in (
            (self.failed_criteria, "failed representability criteria"),
            (self.structurally_failed_criteria, "structurally failed criteria"),
        ):
            values = tuple(item.value for item in criteria)
            if len(values) != len(set(values)) or tuple(sorted(values)) != values:
                raise ValueError(f"{name} must be unique and sorted")
        if not set(self.structurally_failed_criteria) <= set(self.failed_criteria):
            raise ValueError("structural failures must be included in all failed criteria")
        if self.accepted is bool(self.failed_criteria):
            raise ValueError("representability acceptance must exactly reflect failed criteria")
        return self


def _resolve_claim(
    manifest: DatasetManifest,
    reference: ClaimAssessmentReference | DatasetAssessmentReference,
) -> ClaimAssessment:
    matches = tuple(
        assessment
        for assessment in manifest.claim_assessments
        if assessment.assessment_id == reference.assessment_id
    )
    if not matches:
        raise ValueError("representability proof references an unknown claim assessment")
    assessment = matches[0]
    if reference.assessment_fingerprint != assessment.fingerprint:
        raise ValueError("representability proof claim fingerprint does not match the manifest")
    return assessment


def _negative_claims(
    manifest: DatasetManifest,
    proof: RepresentabilityProof,
) -> dict[ScientificClaim, ClaimAssessment]:
    resolved = tuple(
        _resolve_claim(manifest, reference) for reference in proof.negative_claim_assessments
    )
    claims = tuple(assessment.claim for assessment in resolved)
    if len(claims) != len(set(claims)):
        raise ValueError("negative references must bind unique scientific claims")
    if any(
        assessment.status not in {EligibilityStatus.INELIGIBLE, EligibilityStatus.NOT_ASSESSED}
        for assessment in resolved
    ):
        raise ValueError("negative representability claims must be ineligible or not assessed")
    return {assessment.claim: assessment for assessment in resolved}


def _required_negative_claims(
    kind: RepresentabilityProofKind,
) -> frozenset[ScientificClaim]:
    common = {
        ScientificClaim.LINEAGE_FATE,
        ScientificClaim.INTERVENTION_EFFECT,
        ScientificClaim.COUNTERFACTUAL_GENERALIZATION,
    }
    if kind is RepresentabilityProofKind.DESTRUCTIVE_POPULATION:
        return frozenset(
            {
                *common,
                ScientificClaim.POPULATION_DYNAMICS,
                ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS,
            }
        )
    return frozenset(
        {
            *common,
            ScientificClaim.POPULATION_DYNAMICS,
            ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS,
        }
    )


def _source_bindings_are_exact(
    manifest: DatasetManifest,
    proof: RepresentabilityProof,
    positive: ClaimAssessment,
    negatives: dict[ScientificClaim, ClaimAssessment],
) -> None:
    required_source_ids = {
        *manifest.slice_spec.selection_source_ids,
        *positive.evidence_source_ids,
        *(
            source_id
            for assessment in negatives.values()
            for source_id in assessment.evidence_source_ids
        ),
        *(
            source_id
            for trace in proof.criterion_traces
            for locator in trace.evidence_locators
            for source_id in (locator.source_id,)
        ),
        *(
            source_id
            for stage in manifest.slice_spec.selection_stages
            for source_id in stage.source_ids
        ),
    }
    bindings = {binding.source_id: binding.sha256 for binding in proof.source_bindings}
    if set(bindings) != required_source_ids:
        raise ValueError("representability source bindings do not exactly cover proof evidence")
    manifest_hashes = {source.source_id: source.sha256 for source in manifest.sources}
    if any(
        source_id not in manifest_hashes or manifest_hashes[source_id] != sha256
        for source_id, sha256 in bindings.items()
    ):
        raise ValueError("representability source binding does not match manifest source bytes")


def _trace_has_method(
    traces: dict[RepresentabilityCriterion, RepresentabilityCriterionTrace],
    criterion: RepresentabilityCriterion,
    method: RepresentabilityEvidenceMethod,
) -> bool:
    return any(locator.method is method for locator in traces[criterion].evidence_locators)


def verify_representability(
    manifest: DatasetManifest,
    proof: RepresentabilityProof,
) -> RepresentabilityResolution:
    """Verify exact evidence identity and derive a fail-closed representability decision."""

    reference = proof.assessment_reference
    if reference.dataset_manifest_fingerprint != manifest.fingerprint:
        raise ValueError("representability proof does not bind this dataset manifest")
    if proof.slice_fingerprint != manifest.slice_spec.fingerprint:
        raise ValueError("representability proof does not bind this dataset slice")
    positive = _resolve_claim(manifest, reference)
    negatives = _negative_claims(manifest, proof)
    required_negative_claims = _required_negative_claims(proof.proof_kind)
    if set(negatives) != required_negative_claims:
        missing = sorted(item.value for item in required_negative_claims - set(negatives))
        unexpected = sorted(item.value for item in set(negatives) - required_negative_claims)
        raise ValueError(
            "representability proof has an invalid negative-claim set: "
            f"missing={missing}, unexpected={unexpected}"
        )
    _source_bindings_are_exact(manifest, proof, positive, negatives)

    supported = {EligibilityStatus.ELIGIBLE, EligibilityStatus.CONDITIONALLY_ELIGIBLE}
    if positive.status not in supported:
        raise ValueError("representability proof requires a supported positive assessment")

    traces = {trace.criterion: trace for trace in proof.criterion_traces}
    sampling = manifest.experimental_design.sampling
    scope = positive.scope
    modality_by_key = {
        modality.modality.key: modality for modality in manifest.capabilities.modalities
    }
    output_by_id = {
        output.readout_id: output for output in manifest.capabilities.functional.outputs
    }
    scoped_modalities = tuple(modality_by_key[item.key] for item in scope.modalities)
    scoped_outputs = tuple(output_by_id[item] for item in scope.functional_readout_ids)

    structurally_satisfied: dict[RepresentabilityCriterion, bool] = {
        criterion: True for criterion in _REQUIRED_CRITERIA[proof.proof_kind]
    }
    bound_source_ids = {binding.source_id for binding in proof.source_bindings}
    checksum_source_ids = {
        locator.source_id
        for locator in traces[RepresentabilityCriterion.SOURCE_BYTES_BOUND].evidence_locators
        if locator.method is RepresentabilityEvidenceMethod.CHECKSUM_VERIFICATION
    }
    structurally_satisfied[RepresentabilityCriterion.SOURCE_BYTES_BOUND] = (
        checksum_source_ids == bound_source_ids
    )
    exact_slice_methods = {
        locator.method
        for locator in traces[RepresentabilityCriterion.EXACT_SLICE_BOUND].evidence_locators
    }
    required_slice_method = (
        RepresentabilityEvidenceMethod.CHECKSUM_VERIFICATION
        if manifest.slice_spec.kind is DatasetSliceKind.WHOLE_ARTIFACT
        else RepresentabilityEvidenceMethod.SELECTOR_EXECUTION
    )
    structurally_satisfied[RepresentabilityCriterion.EXACT_SLICE_BOUND] = (
        required_slice_method in exact_slice_methods
    )
    if proof.proof_kind is RepresentabilityProofKind.DESTRUCTIVE_POPULATION:
        if positive.claim is not ScientificClaim.SNAPSHOT_STATE_PRIOR:
            raise ValueError("destructive-population proofs must bind a snapshot-state prior")
        structurally_satisfied[RepresentabilityCriterion.POPULATION_SUBJECT] = (
            sampling.subject_kind is SamplingSubjectKind.POPULATION
            and scope.subject_kind is SamplingSubjectKind.POPULATION
            and sampling.subject_unit is not ExperimentalUnitLevel.CELL
        )
        structurally_satisfied[RepresentabilityCriterion.DESTRUCTIVE_COLLECTION] = bool(
            scoped_modalities
        ) and all(modality.destructive for modality in scoped_modalities)
        structurally_satisfied[RepresentabilityCriterion.POPULATION_LINKAGE_BOUNDARY] = (
            scope.system_boundary is SystemBoundary.POPULATION
            and (
                (
                    sampling.mode is SamplingMode.ENDPOINT_DESTRUCTIVE
                    and sampling.linkage is SubjectLinkage.NONE
                )
                or (
                    sampling.mode is SamplingMode.REPEATED_POPULATION_DESTRUCTIVE
                    and sampling.linkage is SubjectLinkage.SAME_POPULATION
                )
            )
        )
        structurally_satisfied[RepresentabilityCriterion.INDIVIDUAL_CAST_REJECTED] = (
            ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS in negatives
        )
    else:
        if positive.claim is not ScientificClaim.FUNCTIONAL_OUTCOME:
            raise ValueError(
                "individual functional-recorder proofs must bind a functional-outcome claim"
            )
        baseline_windows = tuple(
            modality.collection_time_window
            for modality in scoped_modalities
            if modality.collection_time_window is not None
        )
        target_windows = tuple(
            output.measurement_time_window
            for output in scoped_outputs
            if output.measurement_time_window is not None
        )
        structurally_satisfied[RepresentabilityCriterion.INDIVIDUAL_CELL_SUBJECT] = (
            sampling.subject_kind is SamplingSubjectKind.INDIVIDUAL_CELL
            and sampling.subject_unit is ExperimentalUnitLevel.CELL
            and scope.subject_kind is SamplingSubjectKind.INDIVIDUAL_CELL
        )
        structurally_satisfied[RepresentabilityCriterion.SAME_CELL_LINKAGE] = (
            sampling.linkage is SubjectLinkage.SAME_CELL
            and bool(scoped_modalities)
            and all(
                modality.subject_alignment is SubjectAlignment.SAME_CELL
                for modality in scoped_modalities
            )
            and bool(scoped_outputs)
            and all(
                output.subject_alignment is SubjectAlignment.SAME_CELL for output in scoped_outputs
            )
            and _trace_has_method(
                traces,
                RepresentabilityCriterion.SAME_CELL_LINKAGE,
                RepresentabilityEvidenceMethod.DIRECT_IMAGE_TRACKING,
            )
        )
        structurally_satisfied[RepresentabilityCriterion.VIABILITY_PRESERVING_COLLECTION] = (
            sampling.mode
            in {
                SamplingMode.PARTIAL_NONDESTRUCTIVE,
                SamplingMode.LONGITUDINAL_NONDESTRUCTIVE,
            }
            and bool(scoped_modalities)
            and all(not modality.destructive for modality in scoped_modalities)
        )
        same_clock_and_ordered = bool(baseline_windows and target_windows) and all(
            baseline.reference_event == target.reference_event
            and baseline.latest_seconds < target.earliest_seconds
            for baseline in baseline_windows
            for target in target_windows
        )
        cutoff = scope.inference_cutoff_window
        baselines_bind_cutoff = cutoff is not None and any(
            baseline.fingerprint == cutoff.fingerprint for baseline in baseline_windows
        )
        sampling_binds_cutoff = cutoff is not None and sampling.time_window_id == cutoff.window_id
        structurally_satisfied[RepresentabilityCriterion.BASELINE_BEFORE_TARGET] = (
            same_clock_and_ordered and baselines_bind_cutoff and sampling_binds_cutoff
        )
        target_fingerprints = {window.fingerprint for window in scope.horizon_windows}
        structurally_satisfied[RepresentabilityCriterion.FUTURE_SAME_CELL_FUNCTIONAL_READOUT] = (
            bool(target_windows)
            and bool(target_fingerprints)
            and {window.fingerprint for window in target_windows} == target_fingerprints
        )
        structurally_satisfied[RepresentabilityCriterion.POPULATION_CAST_REJECTED] = (
            ScientificClaim.POPULATION_DYNAMICS in negatives
        )

    structurally_satisfied[RepresentabilityCriterion.CLONE_CAST_REJECTED] = (
        ScientificClaim.LINEAGE_FATE in negatives
    )
    structurally_satisfied[RepresentabilityCriterion.CAUSAL_OVERCLAIM_REJECTED] = (
        ScientificClaim.INTERVENTION_EFFECT in negatives
    )
    structurally_satisfied[RepresentabilityCriterion.TRANSPORT_OVERCLAIM_REJECTED] = (
        ScientificClaim.COUNTERFACTUAL_GENERALIZATION in negatives
    )

    structurally_failed = tuple(
        sorted(
            (criterion for criterion, satisfied in structurally_satisfied.items() if not satisfied),
            key=lambda item: item.value,
        )
    )
    reviewed_failed = {
        trace.criterion
        for trace in proof.criterion_traces
        if trace.status is RepresentabilityCriterionStatus.FAILED
    }
    failed = tuple(sorted({*structurally_failed, *reviewed_failed}, key=lambda item: item.value))
    return RepresentabilityResolution(
        proof_fingerprint=proof.fingerprint,
        manifest_fingerprint=manifest.fingerprint,
        proof_kind=proof.proof_kind,
        accepted=not failed,
        failed_criteria=failed,
        structurally_failed_criteria=structurally_failed,
    )
