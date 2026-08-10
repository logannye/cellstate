from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from cellstate.data import (
    AssessmentScope,
    ClaimAssessment,
    ClaimAssessmentReference,
    CohortSelectionStage,
    DatasetAssessmentReference,
    DatasetCapabilities,
    DatasetManifest,
    DatasetSliceKind,
    DatasetSliceSpec,
    DataUseCase,
    DataUsePolicy,
    EligibilityStatus,
    ExperimentalDesign,
    ExperimentalUnitLevel,
    ExperimentalUnitSpec,
    FunctionalCapability,
    FunctionalReadout,
    FunctionalReadoutDerivation,
    IdentificationBasis,
    ModalitySpec,
    PermissionStatus,
    PublicRealDataOrigin,
    ReadoutStatus,
    RepresentabilityCriterion,
    RepresentabilityCriterionStatus,
    RepresentabilityCriterionTrace,
    RepresentabilityEvidenceLocator,
    RepresentabilityEvidenceMethod,
    RepresentabilityProof,
    RepresentabilityProofKind,
    RepresentabilitySourceBinding,
    SamplingDesign,
    SamplingMode,
    SamplingSubjectKind,
    ScientificClaim,
    SourceArtifact,
    SourceKind,
    SubjectAlignment,
    SubjectLinkage,
    TemporalWindow,
    TimingCapability,
    UsePermission,
    canonical_selected_record_ids_sha256,
    verify_representability,
)
from cellstate.domain import OntologyTerm, SystemBoundary


def _term(label: str) -> OntologyTerm:
    return OntologyTerm(label=label)


def _source(source_id: str, kind: SourceKind, hash_character: str) -> SourceArtifact:
    return SourceArtifact(
        source_id=source_id,
        kind=kind,
        uri=f"https://example.org/{source_id}",
        sha256=hash_character * 64,
        media_type="application/octet-stream",
        accession="GSE141064",
        release="2026-08-09",
        byte_count=1024,
        retrieved_at=datetime(2026, 8, 9, tzinfo=UTC),
    )


def _policy(sources: tuple[SourceArtifact, ...]) -> DataUsePolicy:
    return DataUsePolicy(
        policy_id="review-required-policy",
        source_ids=tuple(source.source_id for source in sources),
        license_name="Terms require independent review",
        terms_url="https://example.org/terms",
        reviewed_on=date(2026, 8, 9),
        permissions=tuple(
            UsePermission(
                use_case=use_case,
                status=PermissionStatus.PROHIBITED,
                conditions=("This fixture intentionally denies use.",),
            )
            for use_case in DataUseCase
        ),
    )


def _negative_scope(claim: ScientificClaim) -> AssessmentScope:
    subject_and_boundary = {
        ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS: (
            SamplingSubjectKind.INDIVIDUAL_CELL,
            SystemBoundary.ISOLATED_CELL,
        ),
        ScientificClaim.LINEAGE_FATE: (
            SamplingSubjectKind.CLONE,
            SystemBoundary.CLONE,
        ),
    }.get(claim, (SamplingSubjectKind.POPULATION, SystemBoundary.POPULATION))
    return AssessmentScope(
        subject_kind=subject_and_boundary[0],
        system_boundary=subject_and_boundary[1],
        biological_systems=(_term("cultured human cells"),),
    )


def _negative_assessments() -> tuple[ClaimAssessment, ...]:
    claims = (
        ScientificClaim.COUNTERFACTUAL_GENERALIZATION,
        ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS,
        ScientificClaim.INTERVENTION_EFFECT,
        ScientificClaim.LINEAGE_FATE,
        ScientificClaim.POPULATION_DYNAMICS,
    )
    return tuple(
        ClaimAssessment(
            assessment_id=f"negative-{claim.value}",
            claim=claim,
            status=EligibilityStatus.INELIGIBLE,
            identification_basis=IdentificationBasis.NONE,
            scope=_negative_scope(claim),
            blockers=("The reviewed slice does not identify this claim.",),
        )
        for claim in claims
    )


def _population_manifest() -> DatasetManifest:
    metadata = _source("metadata", SourceKind.METADATA, "a")
    paper = _source("paper", SourceKind.DOCUMENTATION, "b")
    processed = _source("processed", SourceKind.PROCESSED, "c")
    sources = (metadata, paper, processed)
    transcriptome = _term("transcriptome")
    scope = AssessmentScope(
        subject_kind=SamplingSubjectKind.POPULATION,
        system_boundary=SystemBoundary.POPULATION,
        biological_systems=(_term("cultured human cells"),),
        modalities=(transcriptome,),
    )
    positive = ClaimAssessment(
        assessment_id="conditional-snapshot-prior",
        claim=ScientificClaim.SNAPSHOT_STATE_PRIOR,
        status=EligibilityStatus.CONDITIONALLY_ELIGIBLE,
        identification_basis=IdentificationBasis.DESCRIPTIVE,
        scope=scope,
        evidence_source_ids=("metadata", "processed"),
        evidence_notes=("The slice measures a destructive cell-population snapshot.",),
        assumptions=("Use is restricted to the represented K562 population snapshot.",),
    )
    return DatasetManifest(
        dataset_id="k562-destructive-population",
        version="2026-08-09",
        title="K562 destructive population fixture",
        description="Minimal real-data-shaped representability contract fixture.",
        origin=PublicRealDataOrigin(
            repository="public repository",
            study_accession="GSE141064",
            release="2026-08-09",
            species=(_term("Homo sapiens"),),
            biological_systems=(_term("cultured human cells"),),
        ),
        sources=sources,
        slice_spec=DatasetSliceSpec(
            kind=DatasetSliceKind.WHOLE_ARTIFACT,
            slice_id="k562-reviewed-record-axis",
            selection_source_ids=("processed",),
            record_id_field="cell_id",
            selected_record_ids_sha256="d" * 64,
            selected_record_count=100,
            selected_subject_count=1,
        ),
        use_policies=(_policy(sources),),
        experimental_design=ExperimentalDesign(
            units=(
                ExperimentalUnitSpec(
                    level=ExperimentalUnitLevel.CULTURE,
                    id_field="population_id",
                    source_ids=("metadata",),
                ),
                ExperimentalUnitSpec(
                    level=ExperimentalUnitLevel.CELL,
                    id_field="cell_id",
                    source_ids=("processed",),
                    parent_level=ExperimentalUnitLevel.CULTURE,
                ),
            ),
            sampling=SamplingDesign(
                subject_kind=SamplingSubjectKind.POPULATION,
                subject_unit=ExperimentalUnitLevel.CULTURE,
                subject_id_field="population_id",
                source_ids=("metadata",),
                mode=SamplingMode.ENDPOINT_DESTRUCTIVE,
                linkage=SubjectLinkage.NONE,
            ),
            default_split_unit=ExperimentalUnitLevel.CULTURE,
        ),
        capabilities=DatasetCapabilities(
            modalities=(
                ModalitySpec(
                    modality=transcriptome,
                    source_ids=("processed",),
                    subject_alignment=SubjectAlignment.SAME_POPULATION,
                    alignment_group="population-snapshot",
                    alignment_key_field="population_id",
                    processed_available=True,
                    destructive=True,
                ),
            )
        ),
        claim_assessments=(positive, *_negative_assessments()),
    )


def _individual_manifest() -> DatasetManifest:
    metadata = _source("metadata", SourceKind.METADATA, "a")
    paper = _source("paper", SourceKind.DOCUMENTATION, "b")
    transcriptome_source = _source("transcriptome", SourceKind.PROCESSED, "c")
    sources = (metadata, paper, transcriptome_source)
    transcriptome = _term("transcriptome")
    baseline = TemporalWindow(
        window_id="live-seq-pre-lps-window",
        earliest_seconds=-5400.0,
        latest_seconds=-1800.0,
        reference_event="lps-addition",
        source_ids=("paper",),
    )
    target = TemporalWindow(
        window_id="live-seq-mcherry-slope-window",
        earliest_seconds=10800.0,
        latest_seconds=27000.0,
        reference_event="lps-addition",
        source_ids=("metadata", "paper"),
    )
    scope = AssessmentScope(
        subject_kind=SamplingSubjectKind.INDIVIDUAL_CELL,
        system_boundary=SystemBoundary.ISOLATED_CELL,
        biological_systems=(_term("cultured human cells"),),
        modalities=(transcriptome,),
        functional_readout_ids=("mcherry-log-slope",),
        horizon_windows=(target,),
        inference_cutoff_window=baseline,
    )
    positive = ClaimAssessment(
        assessment_id="conditional-same-cell-functional-outcome",
        claim=ScientificClaim.FUNCTIONAL_OUTCOME,
        status=EligibilityStatus.CONDITIONALLY_ELIGIBLE,
        identification_basis=IdentificationBasis.ASSOCIATIONAL,
        scope=scope,
        evidence_source_ids=("metadata", "paper", "transcriptome"),
        evidence_notes=("Live-seq links baseline transcriptome to a later same-cell reporter.",),
        assumptions=("The derived reporter slope is a task-specific functional target.",),
    )
    return DatasetManifest(
        dataset_id="live-seq-functional-recorder",
        version="2026-08-09",
        title="Live-seq individual-cell fixture",
        description="Minimal real-data-shaped functional-recorder contract fixture.",
        origin=PublicRealDataOrigin(
            repository="public repository",
            study_accession="GSE141064",
            release="2026-08-09",
            species=(_term("Homo sapiens"),),
            biological_systems=(_term("cultured human cells"),),
        ),
        sources=sources,
        slice_spec=DatasetSliceSpec(
            kind=DatasetSliceKind.CONTENT_ADDRESSED_SELECTION,
            slice_id="live-seq-raw-g9-functional-recorder",
            selection_source_ids=("metadata",),
            record_id_field="cell_id",
            selected_record_ids_uri="https://example.org/live-seq-17-record-ids.json",
            selected_record_ids_sha256="d" * 64,
            selected_record_count=17,
            selected_subject_count=17,
            selector_id="live-seq-g9-functional-recorder",
            selector_version="1",
            selector_sha256="e" * 64,
            selection_stages=(
                CohortSelectionStage(
                    stage_id="complete-baseline-transcriptomes",
                    input_record_count=40,
                    output_record_count=17,
                    criterion="Retain tracked cells with complete baseline transcriptomes.",
                    source_ids=("metadata",),
                ),
            ),
        ),
        use_policies=(_policy(sources),),
        experimental_design=ExperimentalDesign(
            units=(
                ExperimentalUnitSpec(
                    level=ExperimentalUnitLevel.CULTURE,
                    id_field="culture_id",
                    source_ids=("metadata",),
                ),
                ExperimentalUnitSpec(
                    level=ExperimentalUnitLevel.CELL,
                    id_field="cell_id",
                    source_ids=("metadata",),
                    parent_level=ExperimentalUnitLevel.CULTURE,
                ),
            ),
            sampling=SamplingDesign(
                subject_kind=SamplingSubjectKind.INDIVIDUAL_CELL,
                subject_unit=ExperimentalUnitLevel.CELL,
                subject_id_field="cell_id",
                source_ids=("metadata", "paper"),
                mode=SamplingMode.PARTIAL_NONDESTRUCTIVE,
                linkage=SubjectLinkage.SAME_CELL,
                time_window_id=baseline.window_id,
                attrition_field="complete_baseline_transcriptome",
            ),
            default_split_unit=ExperimentalUnitLevel.CULTURE,
        ),
        capabilities=DatasetCapabilities(
            modalities=(
                ModalitySpec(
                    modality=transcriptome,
                    source_ids=("transcriptome",),
                    subject_alignment=SubjectAlignment.SAME_CELL,
                    alignment_group="tracked-cell",
                    alignment_key_field="cell_id",
                    processed_available=True,
                    destructive=False,
                    collection_time_window=baseline,
                ),
            ),
            timing=TimingCapability(
                source_ids=("metadata", "paper"),
                observation_windows=(baseline, target),
                observation_times_recorded=True,
                event_ordering_recorded=True,
            ),
            functional=FunctionalCapability(
                outputs=(
                    FunctionalReadout(
                        readout_id="mcherry-log-slope",
                        output=_term("mCherry reporter response rate"),
                        source_ids=("metadata", "paper"),
                        value_field="mCherry.log.slope",
                        units="log fluorescence per second",
                        aggregation_level=ExperimentalUnitLevel.CELL,
                        subject_alignment=SubjectAlignment.SAME_CELL,
                        alignment_group="tracked-cell",
                        alignment_key_field="cell_id",
                        status=ReadoutStatus.DERIVED,
                        derivation=FunctionalReadoutDerivation(
                            method_id="linear-reporter-slope",
                            method_version="1",
                            method_sha256="f" * 64,
                            source_value_fields=("mCherry.intensity",),
                            formula="ordinary-least-squares slope of log mCherry intensity",
                        ),
                        measurement_time_window=target,
                    ),
                )
            ),
        ),
        claim_assessments=(positive, *_negative_assessments()),
    )


def _positive_reference(manifest: DatasetManifest) -> DatasetAssessmentReference:
    positive = manifest.claim_assessments[0]
    return DatasetAssessmentReference(
        dataset_manifest_fingerprint=manifest.fingerprint,
        assessment_id=positive.assessment_id,
        assessment_fingerprint=positive.fingerprint,
    )


def _negative_references(manifest: DatasetManifest) -> tuple[ClaimAssessmentReference, ...]:
    return tuple(
        sorted(
            (
                ClaimAssessmentReference(
                    assessment_id=assessment.assessment_id,
                    assessment_fingerprint=assessment.fingerprint,
                )
                for assessment in manifest.claim_assessments[1:]
            ),
            key=lambda reference: reference.assessment_id,
        )
    )


def _criterion_trace(
    criterion: RepresentabilityCriterion,
    *,
    manifest: DatasetManifest,
    individual: bool,
) -> RepresentabilityCriterionTrace:
    source_id = "paper"
    method = RepresentabilityEvidenceMethod.PUBLICATION_METHOD
    if criterion is RepresentabilityCriterion.SOURCE_BYTES_BOUND:
        locators = tuple(
            RepresentabilityEvidenceLocator(
                source_id=source.source_id,
                method=RepresentabilityEvidenceMethod.CHECKSUM_VERIFICATION,
                locator=f"review/checksum/{source.source_id}",
            )
            for source in sorted(manifest.sources, key=lambda item: item.source_id)
        )
    elif criterion is RepresentabilityCriterion.EXACT_SLICE_BOUND:
        source_id = manifest.slice_spec.selection_source_ids[0]
        method = (
            RepresentabilityEvidenceMethod.CHECKSUM_VERIFICATION
            if manifest.slice_spec.kind is DatasetSliceKind.WHOLE_ARTIFACT
            else RepresentabilityEvidenceMethod.SELECTOR_EXECUTION
        )
        locators = (
            RepresentabilityEvidenceLocator(
                source_id=source_id,
                method=method,
                locator=f"review/{criterion.value}",
            ),
        )
    else:
        if individual and criterion is RepresentabilityCriterion.SAME_CELL_LINKAGE:
            method = RepresentabilityEvidenceMethod.DIRECT_IMAGE_TRACKING
        locators = (
            RepresentabilityEvidenceLocator(
                source_id=source_id,
                method=method,
                locator=f"review/{criterion.value}",
            ),
        )
    return RepresentabilityCriterionTrace(
        criterion=criterion,
        status=RepresentabilityCriterionStatus.PASSED,
        evidence_locators=locators,
        evidence_notes=(f"Reviewed evidence passes {criterion.value}.",),
    )


def _proof(manifest: DatasetManifest, kind: RepresentabilityProofKind) -> RepresentabilityProof:
    criteria = (
        (
            RepresentabilityCriterion.EXACT_SLICE_BOUND,
            RepresentabilityCriterion.SOURCE_BYTES_BOUND,
            RepresentabilityCriterion.POPULATION_SUBJECT,
            RepresentabilityCriterion.DESTRUCTIVE_COLLECTION,
            RepresentabilityCriterion.POPULATION_LINKAGE_BOUNDARY,
            RepresentabilityCriterion.INDIVIDUAL_CAST_REJECTED,
            RepresentabilityCriterion.CLONE_CAST_REJECTED,
            RepresentabilityCriterion.CAUSAL_OVERCLAIM_REJECTED,
            RepresentabilityCriterion.TRANSPORT_OVERCLAIM_REJECTED,
        )
        if kind is RepresentabilityProofKind.DESTRUCTIVE_POPULATION
        else (
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
        )
    )
    traces = tuple(
        sorted(
            (
                _criterion_trace(
                    criterion,
                    manifest=manifest,
                    individual=(
                        kind is RepresentabilityProofKind.INDIVIDUAL_CELL_FUNCTIONAL_RECORDER
                    ),
                )
                for criterion in criteria
            ),
            key=lambda trace: trace.criterion.value,
        )
    )
    hashes = {source.source_id: source.sha256 for source in manifest.sources}
    return RepresentabilityProof(
        proof_id=f"{manifest.dataset_id}-review",
        proof_kind=kind,
        assessment_reference=_positive_reference(manifest),
        slice_fingerprint=manifest.slice_spec.fingerprint,
        source_bindings=tuple(
            RepresentabilitySourceBinding(source_id=source_id, sha256=hashes[source_id])
            for source_id in sorted(hashes)
        ),
        criterion_traces=traces,
        negative_claim_assessments=_negative_references(manifest),
        reviewed_by=("cellstate-maintainers",),
        reviewed_on=date(2026, 8, 9),
    )


def _revalidate_proof(proof: RepresentabilityProof, **updates: object) -> RepresentabilityProof:
    payload = proof.model_dump(mode="python")
    payload.update(updates)
    return RepresentabilityProof.model_validate(payload)


def test_selected_record_membership_hash_is_semantic_and_canonical() -> None:
    expected = canonical_selected_record_ids_sha256(("cell-a", "cell-b"))
    assert canonical_selected_record_ids_sha256(("cell-b", "cell-a")) == expected
    with pytest.raises(ValueError, match="unique"):
        canonical_selected_record_ids_sha256(("cell-a", "cell-a"))
    with pytest.raises(ValueError, match="canonical"):
        canonical_selected_record_ids_sha256((" cell-a",))
    with pytest.raises(TypeError, match="not one string"):
        canonical_selected_record_ids_sha256("cell-a")


def test_slice_contract_is_explicit_and_content_addressed() -> None:
    manifest = _individual_manifest()
    assert manifest.schema_version == "0.3-experimental"
    assert manifest.slice_spec.selected_record_count == 17
    assert manifest.slice_spec.selection_stages[0].input_record_count == 40
    payload = manifest.slice_spec.model_dump(mode="python")
    payload["selected_record_ids_uri"] = None
    selector_only = DatasetSliceSpec.model_validate(payload)
    assert selector_only.selected_record_ids_uri is None
    assert selector_only.fingerprint != manifest.slice_spec.fingerprint
    payload = manifest.slice_spec.model_dump(mode="python")
    payload["selector_sha256"] = None
    with pytest.raises(ValidationError, match="exact selector identity"):
        DatasetSliceSpec.model_validate(payload)
    payload = manifest.slice_spec.model_dump(mode="python")
    payload["selected_subject_count"] = 18
    with pytest.raises(ValidationError, match="cannot exceed"):
        DatasetSliceSpec.model_validate(payload)

    manifest_payload = manifest.model_dump(mode="python")
    del manifest_payload["slice_spec"]
    with pytest.raises(ValidationError, match="Field required"):
        DatasetManifest.model_validate(manifest_payload)
    manifest_payload = manifest.model_dump(mode="python")
    manifest_payload["slice_spec"]["selection_source_ids"] = ("unknown-source",)
    manifest_payload["slice_spec"]["selection_stages"][0]["source_ids"] = ("unknown-source",)
    with pytest.raises(ValidationError, match="unknown source artifacts"):
        DatasetManifest.model_validate(manifest_payload)


def test_temporal_intervals_are_preserved_and_never_midpoint_coerced() -> None:
    manifest = _individual_manifest()
    baseline = manifest.capabilities.modalities[0].collection_time_window
    assert baseline is not None
    assert (baseline.earliest_seconds, baseline.latest_seconds) == (-5400.0, -1800.0)
    with pytest.raises(ValidationError, match="bounds must be ordered"):
        TemporalWindow(
            window_id="reversed",
            earliest_seconds=1.0,
            latest_seconds=-1.0,
            reference_event="lps-addition",
            source_ids=("paper",),
        )
    target = manifest.capabilities.functional.outputs[0].measurement_time_window
    assert target is not None
    with pytest.raises(ValidationError, match="must occur after"):
        AssessmentScope(
            subject_kind=SamplingSubjectKind.INDIVIDUAL_CELL,
            system_boundary=SystemBoundary.ISOLATED_CELL,
            biological_systems=(_term("cultured human cells"),),
            horizon_windows=(baseline,),
            inference_cutoff_window=target,
        )


def test_destructive_population_proof_is_go_but_never_legal_authorization() -> None:
    manifest = _population_manifest()
    proof = _proof(manifest, RepresentabilityProofKind.DESTRUCTIVE_POPULATION)
    resolution = verify_representability(manifest, proof)
    assert resolution.accepted
    assert resolution.failed_criteria == ()
    assert resolution.use_permission_evaluated is False
    assert resolution.use_authorized is False
    assert (
        manifest.permission_status(DataUseCase.RESEARCH_MODEL_TRAINING)
        is PermissionStatus.PROHIBITED
    )


def test_exact_source_hash_and_negative_claim_bindings_fail_closed() -> None:
    manifest = _population_manifest()
    proof = _proof(manifest, RepresentabilityProofKind.DESTRUCTIVE_POPULATION)
    bindings = list(proof.source_bindings)
    bindings[0] = RepresentabilitySourceBinding(
        source_id=bindings[0].source_id,
        sha256="0" * 64,
    )
    tampered = _revalidate_proof(proof, source_bindings=tuple(bindings))
    with pytest.raises(ValueError, match="source binding"):
        verify_representability(manifest, tampered)

    missing_negative = _revalidate_proof(
        proof,
        negative_claim_assessments=proof.negative_claim_assessments[:-1],
    )
    with pytest.raises(ValueError, match="negative-claim set"):
        verify_representability(manifest, missing_negative)


def test_reviewed_failure_produces_no_go_without_invalidating_the_artifact() -> None:
    manifest = _population_manifest()
    proof = _proof(manifest, RepresentabilityProofKind.DESTRUCTIVE_POPULATION)
    traces = tuple(
        RepresentabilityCriterionTrace(
            criterion=trace.criterion,
            status=RepresentabilityCriterionStatus.FAILED,
            evidence_locators=trace.evidence_locators,
            evidence_notes=trace.evidence_notes,
            blockers=("Reviewer could not establish destructive collection.",),
        )
        if trace.criterion is RepresentabilityCriterion.DESTRUCTIVE_COLLECTION
        else trace
        for trace in proof.criterion_traces
    )
    reviewed_no_go = _revalidate_proof(proof, criterion_traces=traces)
    resolution = verify_representability(manifest, reviewed_no_go)
    assert not resolution.accepted
    assert resolution.failed_criteria == (RepresentabilityCriterion.DESTRUCTIVE_COLLECTION,)
    assert resolution.structurally_failed_criteria == ()


def test_individual_functional_recorder_proof_requires_direct_same_cell_evidence() -> None:
    manifest = _individual_manifest()
    proof = _proof(
        manifest,
        RepresentabilityProofKind.INDIVIDUAL_CELL_FUNCTIONAL_RECORDER,
    )
    assert verify_representability(manifest, proof).accepted

    traces = tuple(
        RepresentabilityCriterionTrace(
            criterion=trace.criterion,
            status=trace.status,
            evidence_locators=tuple(
                RepresentabilityEvidenceLocator(
                    source_id=locator.source_id,
                    method=RepresentabilityEvidenceMethod.PUBLICATION_METHOD,
                    locator=locator.locator,
                )
                for locator in trace.evidence_locators
            ),
            evidence_notes=trace.evidence_notes,
            blockers=trace.blockers,
        )
        if trace.criterion is RepresentabilityCriterion.SAME_CELL_LINKAGE
        else trace
        for trace in proof.criterion_traces
    )
    untyped_linkage = _revalidate_proof(proof, criterion_traces=traces)
    resolution = verify_representability(manifest, untyped_linkage)
    assert not resolution.accepted
    assert resolution.structurally_failed_criteria == (RepresentabilityCriterion.SAME_CELL_LINKAGE,)
