from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cellstate.data import (
    DatasetAssessmentReference,
    DatasetManifest,
    DataUseCase,
    EligibilityStatus,
    PermissionStatus,
    RepresentabilityCriterion,
    RepresentabilityProof,
    SamplingMode,
    SamplingSubjectKind,
    ScientificClaim,
    SubjectAlignment,
    SubjectLinkage,
    canonical_selected_record_ids_sha256,
    verify_representability,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPO_ROOT / "data_manifests" / "reviewed" / "gse141064-raw-g9-functional-recorder.json"
)
PROOF_PATH = REPO_ROOT / "data_manifests" / "proofs" / "gse141064-longitudinal-individual.json"
MEMBERSHIP_PATH = (
    REPO_ROOT / "data_manifests" / "slices" / "gse141064-raw-g9-functional-recorder-record-ids.json"
)


def load_manifest() -> DatasetManifest:
    return DatasetManifest.model_validate_json(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_proof() -> RepresentabilityProof:
    return RepresentabilityProof.model_validate_json(PROOF_PATH.read_text(encoding="utf-8"))


def test_liveseq_membership_is_the_exact_content_addressed_qc17_slice() -> None:
    manifest = load_manifest()
    record_ids = json.loads(MEMBERSHIP_PATH.read_text(encoding="utf-8"))

    assert record_ids == [
        "sample372",
        "sample378",
        "sample386",
        "sample404",
        "sample405",
        "sample416",
        "sample421",
        "sample441",
        "sample452",
        "sample454",
        "sample459",
        "sample466",
        "sample468",
        "sample486",
        "sample498",
        "sample499",
        "sample505",
    ]
    assert record_ids == sorted(record_ids)
    assert len(record_ids) == manifest.slice_spec.selected_record_count == 17
    assert manifest.slice_spec.selected_subject_count == 17
    assert canonical_selected_record_ids_sha256(record_ids) == (
        manifest.slice_spec.selected_record_ids_sha256
    )
    assert manifest.slice_spec.selected_record_ids_sha256 == (
        "2e26c9f32124bc5b92cc1dbc281189f44968d95a86d6e202e2475c000bddf8ff"
    )
    assert manifest.slice_spec.selector_sha256 == (
        "1196d1fc8478623bc1405701062b335df12c205777b3bafa7a2a8360dbb0c1a3"
    )
    assert [
        (stage.input_record_count, stage.output_record_count)
        for stage in manifest.slice_spec.selection_stages
    ] == [(40, 17)]


def test_liveseq_manifest_preserves_same_cell_windows_and_narrow_claim() -> None:
    manifest = load_manifest()
    sampling = manifest.experimental_design.sampling
    modality = manifest.capabilities.modalities[0]
    readout = manifest.capabilities.functional.outputs[0]
    positive = next(
        assessment
        for assessment in manifest.claim_assessments
        if assessment.assessment_id == "functional-outcome-same-cell-recorder"
    )

    assert sampling.subject_kind is SamplingSubjectKind.INDIVIDUAL_CELL
    assert sampling.mode is SamplingMode.PARTIAL_NONDESTRUCTIVE
    assert sampling.linkage is SubjectLinkage.SAME_CELL
    assert sampling.time_window_id == "live-seq-pre-lps-window"
    assert modality.subject_alignment is SubjectAlignment.SAME_CELL
    assert modality.destructive is False
    assert modality.collection_time_window is not None
    assert (
        modality.collection_time_window.earliest_seconds,
        modality.collection_time_window.latest_seconds,
    ) == (-5400.0, -1800.0)
    assert readout.subject_alignment is SubjectAlignment.SAME_CELL
    assert readout.value_field == "mCherry.log.slope"
    assert readout.measurement_time_window is not None
    assert (
        readout.measurement_time_window.earliest_seconds,
        readout.measurement_time_window.latest_seconds,
    ) == (10800.0, 27000.0)
    assert readout.derivation is not None
    assert readout.derivation.method_sha256 == (
        "cfaa5c013874dfc435158b77f557bbc07fcf9673d265c8e33b31fb6750e0e94a"
    )

    assert positive.claim is ScientificClaim.FUNCTIONAL_OUTCOME
    assert positive.status is EligibilityStatus.CONDITIONALLY_ELIGIBLE
    assert positive.scope.inference_cutoff_window == modality.collection_time_window
    assert positive.scope.horizon_windows == (readout.measurement_time_window,)

    negative_statuses = {
        assessment.claim: assessment.status
        for assessment in manifest.claim_assessments
        if assessment.claim
        in {
            ScientificClaim.POPULATION_DYNAMICS,
            ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS,
            ScientificClaim.LINEAGE_FATE,
            ScientificClaim.INTERVENTION_EFFECT,
            ScientificClaim.COUNTERFACTUAL_GENERALIZATION,
            ScientificClaim.RETROSPECTIVE_INTERVENTION_SELECTION,
        }
    }
    assert negative_statuses == {
        ScientificClaim.POPULATION_DYNAMICS: EligibilityStatus.INELIGIBLE,
        ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS: EligibilityStatus.NOT_ASSESSED,
        ScientificClaim.LINEAGE_FATE: EligibilityStatus.INELIGIBLE,
        ScientificClaim.INTERVENTION_EFFECT: EligibilityStatus.INELIGIBLE,
        ScientificClaim.COUNTERFACTUAL_GENERALIZATION: EligibilityStatus.INELIGIBLE,
        ScientificClaim.RETROSPECTIVE_INTERVENTION_SELECTION: EligibilityStatus.INELIGIBLE,
    }
    assert manifest.loss_assessments
    assert manifest.metric_assessments
    assert all(
        assessment.status is EligibilityStatus.NOT_ASSESSED
        for assessment in (*manifest.loss_assessments, *manifest.metric_assessments)
    )


def test_liveseq_representability_proof_passes_without_authorizing_use() -> None:
    manifest = load_manifest()
    proof = load_proof()
    resolution = verify_representability(manifest, proof)

    assert resolution.accepted is True
    assert resolution.failed_criteria == ()
    assert resolution.structurally_failed_criteria == ()
    assert resolution.use_permission_evaluated is False
    assert resolution.use_authorized is False

    same_cell_trace = next(
        trace
        for trace in proof.criterion_traces
        if trace.criterion is RepresentabilityCriterion.SAME_CELL_LINKAGE
    )
    assert {locator.method.value for locator in same_cell_trace.evidence_locators} == {
        "direct_image_tracking"
    }


def test_liveseq_geo_permission_remains_unknown_despite_scientific_proof() -> None:
    manifest = load_manifest()
    positive = next(
        assessment
        for assessment in manifest.claim_assessments
        if assessment.assessment_id == "functional-outcome-same-cell-recorder"
    )
    reference = DatasetAssessmentReference(
        dataset_manifest_fingerprint=manifest.fingerprint,
        assessment_id=positive.assessment_id,
        assessment_fingerprint=positive.fingerprint,
    )

    assert (
        manifest.permission_status(
            DataUseCase.RESEARCH_MODEL_TRAINING,
            source_ids=positive.evidence_source_ids,
        )
        is PermissionStatus.UNKNOWN
    )
    resolution = manifest.resolve_assessment(
        reference,
        use_case=DataUseCase.RESEARCH_MODEL_TRAINING,
    )
    assert resolution.scientific_status is EligibilityStatus.CONDITIONALLY_ELIGIBLE
    assert resolution.effective_permission is not None
    assert resolution.effective_permission.status is PermissionStatus.UNKNOWN
    assert resolution.workflow_status is EligibilityStatus.NOT_ASSESSED
    assert resolution.use_allowed_without_additional_review is False

    policy_by_id = {policy.policy_id: policy for policy in manifest.use_policies}
    assert policy_by_id["gse141064-geo-rights-unresolved"].spdx_identifier is None
    assert policy_by_id["liveseq-analysis-gpl-3.0"].spdx_identifier == "GPL-3.0-only"
    assert policy_by_id["liveseq-paper-cc-by-4.0"].spdx_identifier == "CC-BY-4.0"


def test_liveseq_proof_fails_closed_without_direct_image_tracking() -> None:
    manifest = load_manifest()
    payload = json.loads(PROOF_PATH.read_text(encoding="utf-8"))
    same_cell = next(
        trace for trace in payload["criterion_traces"] if trace["criterion"] == "same_cell_linkage"
    )
    same_cell["evidence_locators"][0]["method"] = "publication_method"
    proof = RepresentabilityProof.model_validate_json(json.dumps(payload))

    resolution = verify_representability(manifest, proof)
    assert resolution.accepted is False
    assert resolution.failed_criteria == (RepresentabilityCriterion.SAME_CELL_LINKAGE,)
    assert resolution.structurally_failed_criteria == (RepresentabilityCriterion.SAME_CELL_LINKAGE,)


def test_liveseq_proof_rejects_a_source_hash_not_bound_by_the_manifest() -> None:
    manifest = load_manifest()
    payload = json.loads(PROOF_PATH.read_text(encoding="utf-8"))
    payload["source_bindings"][0]["sha256"] = "0" * 64
    proof = RepresentabilityProof.model_validate_json(json.dumps(payload))

    with pytest.raises(ValueError, match="does not match manifest source bytes"):
        verify_representability(manifest, proof)


def test_liveseq_manifest_rejects_relabeling_the_17_id_fixture_as_24_records() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["slice_spec"]["selected_record_count"] = 24

    with pytest.raises(ValidationError, match="final cohort-selection stage"):
        DatasetManifest.model_validate_json(json.dumps(payload))
