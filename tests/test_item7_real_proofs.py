from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from cellstate.data import (
    DatasetManifest,
    DataUseCase,
    EligibilityStatus,
    ExperimentalUnitLevel,
    PermissionStatus,
    RepresentabilityCriterion,
    RepresentabilityProof,
    SamplingMode,
    SamplingSubjectKind,
    ScientificClaim,
    SubjectLinkage,
    verify_representability,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data_manifests/reviewed/replogle-2022-k562-essential.json"
PROOF_PATH = ROOT / "data_manifests/proofs/replogle-2022-k562-destructive-population.json"


def load_review() -> tuple[DatasetManifest, RepresentabilityProof]:
    manifest = DatasetManifest.model_validate_json(MANIFEST_PATH.read_text(encoding="utf-8"))
    proof = RepresentabilityProof.model_validate_json(PROOF_PATH.read_text(encoding="utf-8"))
    return manifest, proof


def test_replogle_exact_public_sources_slice_and_license_are_frozen() -> None:
    manifest, _ = load_review()
    sources = {source.source_id: source for source in manifest.sources}
    h5ad = sources["replogle-k562-essential-h5ad"]
    metadata = sources["figshare-20029387-v1-metadata"]

    assert h5ad.uri.endswith("ReplogleWeissman2022_K562_essential.h5ad/content")
    assert h5ad.byte_count == 1_546_729_675
    assert h5ad.sha256 == "412fd0df8c4ccea9f4db91cd88033c49200838b29d40945e48574be588b48789"
    assert metadata.byte_count == 11_402
    assert metadata.sha256 == ("4d7abba9bdedf8b8484aab0aacdce9fbbb21b32fb10b6ef62c633385449f8968")

    slice_spec = manifest.slice_spec
    assert slice_spec.kind.value == "whole_artifact"
    assert slice_spec.record_id_field == "cell_barcode"
    assert slice_spec.selected_record_count == 310_385
    assert slice_spec.selected_subject_count == 1
    assert slice_spec.selected_record_ids_uri is None
    assert slice_spec.selected_record_ids_sha256 == (
        "27c981a89a625aa9c3245a9eda97db44df17f7c22d4a91c7ecd906880b6d3546"
    )

    assert len(manifest.use_policies) == 1
    policy = manifest.use_policies[0]
    assert policy.spdx_identifier == "CC-BY-4.0"
    assert {permission.use_case for permission in policy.permissions} == set(DataUseCase)
    assert all(permission.status is PermissionStatus.PERMITTED for permission in policy.permissions)
    assert policy.attribution_requirements


def test_replogle_population_unit_and_destructive_semantics_do_not_invent_replicates() -> None:
    manifest, _ = load_review()
    design = manifest.experimental_design
    sampling = design.sampling
    units = {unit.level: unit for unit in design.units}

    assert sampling.subject_kind is SamplingSubjectKind.POPULATION
    assert sampling.subject_unit is ExperimentalUnitLevel.CULTURE
    assert sampling.mode is SamplingMode.ENDPOINT_DESTRUCTIVE
    assert sampling.linkage is SubjectLinkage.NONE
    assert design.default_split_unit is ExperimentalUnitLevel.CULTURE
    assert design.biological_replicate_unit is None
    assert design.randomization_unit is None
    assert units[ExperimentalUnitLevel.SAMPLE].id_field == "batch"
    assert units[ExperimentalUnitLevel.SAMPLE].parent_level is ExperimentalUnitLevel.CULTURE
    assert design.batch_fields == ("batch",)
    assert all(modality.destructive for modality in manifest.capabilities.modalities)
    assert manifest.capabilities.timing.timepoints_seconds == (518_400.0,)


def test_replogle_ledger_admits_only_a_conditional_descriptive_snapshot_claim() -> None:
    manifest, proof = load_review()
    by_claim = {assessment.claim: assessment for assessment in manifest.claim_assessments}
    supported = {
        EligibilityStatus.ELIGIBLE,
        EligibilityStatus.CONDITIONALLY_ELIGIBLE,
    }

    assert set(by_claim) == set(ScientificClaim)
    assert [
        assessment.claim
        for assessment in manifest.claim_assessments
        if assessment.status in supported
    ] == [ScientificClaim.SNAPSHOT_STATE_PRIOR]
    snapshot = by_claim[ScientificClaim.SNAPSHOT_STATE_PRIOR]
    assert snapshot.status is EligibilityStatus.CONDITIONALLY_ELIGIBLE
    assert snapshot.identification_basis.value == "descriptive"
    assert len(snapshot.assumptions) == 3
    assert by_claim[ScientificClaim.ASSAY_MEASUREMENT_MODEL].status is (
        EligibilityStatus.NOT_ASSESSED
    )
    assert all(
        by_claim[claim].status is EligibilityStatus.INELIGIBLE
        for claim in set(ScientificClaim)
        - {ScientificClaim.SNAPSHOT_STATE_PRIOR, ScientificClaim.ASSAY_MEASUREMENT_MODEL}
    )
    assert manifest.loss_assessments == ()
    assert manifest.metric_assessments == ()

    workflow = manifest.resolve_assessment(
        proof.assessment_reference,
        use_case=DataUseCase.RESEARCH_MODEL_TRAINING,
    )
    assert workflow.scientific_status is EligibilityStatus.CONDITIONALLY_ELIGIBLE
    assert workflow.effective_permission is not None
    assert workflow.effective_permission.status is PermissionStatus.PERMITTED
    assert workflow.workflow_status is EligibilityStatus.CONDITIONALLY_ELIGIBLE
    assert workflow.use_allowed_without_additional_review is False


def test_replogle_destructive_population_proof_accepts_without_authorizing_use() -> None:
    manifest, proof = load_review()
    resolution = verify_representability(manifest, proof)

    assert manifest.fingerprint == (
        "63257e3c13652e6052a71481a66c6b1ee95da360a9cfa67555215c5bf2c82881"
    )
    assert sha256(manifest.canonical_json_bytes).hexdigest() == manifest.fingerprint
    assert manifest.slice_spec.fingerprint == (
        "1998ee5e4c3b5173f583b2f680853e96d0b9b8ba56dc90889762c3c4896b9c68"
    )
    assert proof.assessment_reference.assessment_fingerprint == (
        "7db8ab830ad1b83b9876cccf5145aa5f68060d06c026b2b55fc59b16ad65ec2f"
    )
    assert resolution.accepted is True
    assert resolution.failed_criteria == ()
    assert resolution.structurally_failed_criteria == ()
    assert resolution.selector_execution_replayed is False
    assert resolution.source_bytes_resolved is False
    assert resolution.use_permission_evaluated is False
    assert resolution.use_authorized is False
    assert {trace.criterion for trace in proof.criterion_traces} == {
        RepresentabilityCriterion.CAUSAL_OVERCLAIM_REJECTED,
        RepresentabilityCriterion.CLONE_CAST_REJECTED,
        RepresentabilityCriterion.DESTRUCTIVE_COLLECTION,
        RepresentabilityCriterion.EXACT_SLICE_BOUND,
        RepresentabilityCriterion.INDIVIDUAL_CAST_REJECTED,
        RepresentabilityCriterion.POPULATION_LINKAGE_BOUNDARY,
        RepresentabilityCriterion.POPULATION_SUBJECT,
        RepresentabilityCriterion.SOURCE_BYTES_BOUND,
        RepresentabilityCriterion.TRANSPORT_OVERCLAIM_REJECTED,
    }


def test_replogle_proof_fails_closed_on_content_or_reference_tampering() -> None:
    manifest, proof = load_review()
    bad_slice = proof.model_copy(update={"slice_fingerprint": "0" * 64})
    with pytest.raises(ValueError, match="does not bind this dataset slice"):
        verify_representability(manifest, bad_slice)

    bad_binding = proof.source_bindings[0].model_copy(update={"sha256": "0" * 64})
    bad_sources = proof.model_copy(
        update={"source_bindings": (bad_binding, *proof.source_bindings[1:])}
    )
    with pytest.raises(ValueError, match="does not match manifest source bytes"):
        verify_representability(manifest, bad_sources)

    first_negative = proof.negative_claim_assessments[0]
    bad_negative = first_negative.model_copy(update={"assessment_fingerprint": "0" * 64})
    bad_claims = proof.model_copy(
        update={
            "negative_claim_assessments": (
                bad_negative,
                *proof.negative_claim_assessments[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="claim fingerprint does not match"):
        verify_representability(manifest, bad_claims)


def test_replogle_review_commits_contracts_not_large_biology_bytes() -> None:
    assert MANIFEST_PATH.stat().st_size < 25_000
    assert PROOF_PATH.stat().st_size < 15_000
    combined = MANIFEST_PATH.read_text(encoding="utf-8") + PROOF_PATH.read_text(encoding="utf-8")
    assert "/Volumes/" not in combined
    assert "selected_record_ids_uri" not in combined
