from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from cellstate.data import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    AssessmentKind,
    AssessmentScope,
    AssignmentMechanism,
    ClaimAssessment,
    ClaimAssessmentReference,
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
    IdentificationBasis,
    InterventionCapability,
    LossEligibilityAssessment,
    MetricEligibilityAssessment,
    MetricFamily,
    MetricPartitionPurpose,
    ModalitySpec,
    PermissionStatus,
    PublicRealDataOrigin,
    ReadoutStatus,
    SamplingDesign,
    SamplingMode,
    SamplingSubjectKind,
    ScientificClaim,
    SourceArtifact,
    SourceKind,
    SubjectAlignment,
    SubjectLinkage,
    TimingCapability,
    UsePermission,
)
from cellstate.domain import OntologyTerm, SystemBoundary
from cellstate.training import LossKind


def term(label: str) -> OntologyTerm:
    return OntologyTerm(label=label)


def source(source_id: str, kind: SourceKind) -> SourceArtifact:
    return SourceArtifact(
        source_id=source_id,
        kind=kind,
        uri=f"https://example.org/{source_id}.bin",
        sha256="a" * 64,
        media_type="application/octet-stream",
        accession="GSE-V02",
        release="2026-08-09",
        byte_count=1024,
        retrieved_at=datetime(2026, 8, 9, tzinfo=UTC),
    )


def permissions(
    *,
    benchmark: PermissionStatus = PermissionStatus.PERMITTED,
) -> tuple[UsePermission, ...]:
    return tuple(
        UsePermission(
            use_case=use_case,
            status=(
                benchmark
                if use_case is DataUseCase.BENCHMARK_EVALUATION
                else PermissionStatus.PERMITTED
            ),
            conditions=("Benchmark use is prohibited by the repository layer.",)
            if use_case is DataUseCase.BENCHMARK_EVALUATION
            and benchmark is not PermissionStatus.PERMITTED
            else (),
        )
        for use_case in DataUseCase
    )


def population_scope(
    *,
    horizon: float,
    modalities: tuple[OntologyTerm, ...],
) -> AssessmentScope:
    return AssessmentScope(
        subject_kind=SamplingSubjectKind.POPULATION,
        system_boundary=SystemBoundary.POPULATION,
        biological_systems=(term("cultured cell population"),),
        modalities=modalities,
        horizons_seconds=(horizon,),
        inference_cutoff_seconds=0.0,
    )


def functional_scope(*, horizon: float, readout_id: str) -> AssessmentScope:
    return AssessmentScope(
        subject_kind=SamplingSubjectKind.POPULATION,
        system_boundary=SystemBoundary.POPULATION,
        biological_systems=(term("cultured cell population"),),
        modalities=(term("transcriptome"),),
        functional_readout_ids=(readout_id,),
        horizons_seconds=(horizon,),
        inference_cutoff_seconds=0.0,
    )


def claim(
    assessment_id: str,
    scientific_claim: ScientificClaim,
    scope: AssessmentScope,
    *,
    status: EligibilityStatus = EligibilityStatus.ELIGIBLE,
) -> ClaimAssessment:
    return ClaimAssessment(
        assessment_id=assessment_id,
        claim=scientific_claim,
        status=status,
        identification_basis=IdentificationBasis.ASSOCIATIONAL,
        scope=scope,
        evidence_source_ids=("processed", "metadata"),
        evidence_notes=(
            "Exact processed measurements and experimental metadata support this role.",
        ),
        assumptions=("Exchangeability holds inside the exact assessment scope.",)
        if status is EligibilityStatus.CONDITIONALLY_ELIGIBLE
        else (),
    )


def claim_reference(assessment: ClaimAssessment) -> ClaimAssessmentReference:
    return ClaimAssessmentReference(
        assessment_id=assessment.assessment_id,
        assessment_fingerprint=assessment.fingerprint,
    )


def manifest_factory() -> DatasetManifest:
    raw = source("raw", SourceKind.RAW)
    processed = source("processed", SourceKind.PROCESSED)
    metadata = source("metadata", SourceKind.METADATA)
    documentation = source("documentation", SourceKind.DOCUMENTATION)
    transcriptome = term("transcriptome")
    surface_proteome = term("surface proteome")
    viability = term("cell viability")
    design = ExperimentalDesign(
        units=(
            ExperimentalUnitSpec(
                level=ExperimentalUnitLevel.WELL,
                id_field="well_id",
                source_ids=("metadata",),
            ),
            ExperimentalUnitSpec(
                level=ExperimentalUnitLevel.SAMPLE,
                id_field="sample_id",
                source_ids=("metadata",),
                parent_level=ExperimentalUnitLevel.WELL,
            ),
            ExperimentalUnitSpec(
                level=ExperimentalUnitLevel.CELL,
                id_field="cell_id",
                source_ids=("processed",),
                parent_level=ExperimentalUnitLevel.SAMPLE,
            ),
        ),
        sampling=SamplingDesign(
            subject_kind=SamplingSubjectKind.POPULATION,
            subject_unit=ExperimentalUnitLevel.SAMPLE,
            subject_id_field="sample_id",
            source_ids=("metadata",),
            mode=SamplingMode.REPEATED_POPULATION_DESTRUCTIVE,
            linkage=SubjectLinkage.SAME_POPULATION,
            time_field="time_seconds",
            source_time_units="s",
        ),
        default_split_unit=ExperimentalUnitLevel.WELL,
        biological_replicate_unit=ExperimentalUnitLevel.WELL,
    )
    capabilities = DatasetCapabilities(
        modalities=(
            ModalitySpec(
                modality=transcriptome,
                source_ids=("raw", "processed"),
                subject_alignment=SubjectAlignment.SAME_SAMPLE,
                alignment_group="sample-omics",
                alignment_key_field="sample_id",
                raw_available=True,
                processed_available=True,
                destructive=True,
            ),
            ModalitySpec(
                modality=surface_proteome,
                source_ids=("processed",),
                subject_alignment=SubjectAlignment.SAME_SAMPLE,
                alignment_group="sample-omics",
                alignment_key_field="sample_id",
                processed_available=True,
                destructive=True,
            ),
        ),
        timing=TimingCapability(
            source_ids=("metadata",),
            timepoints_seconds=(0.0, 86400.0, 172800.0),
            observation_times_recorded=True,
        ),
        functional=FunctionalCapability(
            outputs=(
                FunctionalReadout(
                    readout_id="viability-24h",
                    output=viability,
                    source_ids=("processed",),
                    value_field="viability_fraction_24h",
                    units="fraction",
                    aggregation_level=ExperimentalUnitLevel.WELL,
                    subject_alignment=SubjectAlignment.SAME_POPULATION,
                    alignment_group="well-function",
                    alignment_key_field="well_id",
                    status=ReadoutStatus.DIRECT,
                    measurement_time_seconds=86400.0,
                ),
                FunctionalReadout(
                    readout_id="viability-48h",
                    output=viability,
                    source_ids=("processed",),
                    value_field="viability_fraction_48h",
                    units="fraction",
                    aggregation_level=ExperimentalUnitLevel.WELL,
                    subject_alignment=SubjectAlignment.SAME_POPULATION,
                    alignment_group="well-function",
                    alignment_key_field="well_id",
                    status=ReadoutStatus.DIRECT,
                    measurement_time_seconds=172800.0,
                ),
            )
        ),
    )
    pop_24 = claim(
        "claim-population-24h",
        ScientificClaim.POPULATION_DYNAMICS,
        population_scope(horizon=86400.0, modalities=(transcriptome, surface_proteome)),
    )
    pop_48 = claim(
        "claim-population-48h",
        ScientificClaim.POPULATION_DYNAMICS,
        population_scope(horizon=172800.0, modalities=(transcriptome, surface_proteome)),
    )
    function_24 = claim(
        "claim-function-24h",
        ScientificClaim.FUNCTIONAL_OUTCOME,
        functional_scope(horizon=86400.0, readout_id="viability-24h"),
    )
    function_48 = claim(
        "claim-function-48h",
        ScientificClaim.FUNCTIONAL_OUTCOME,
        functional_scope(horizon=172800.0, readout_id="viability-48h"),
    )
    loss = LossEligibilityAssessment(
        assessment_id="loss-future-24h",
        loss_kind=LossKind.MULTI_HORIZON_FUTURE,
        status=EligibilityStatus.ELIGIBLE,
        scope=pop_24.scope,
        required_split_unit=ExperimentalUnitLevel.WELL,
        data_source_ids=("processed", "metadata"),
        supporting_claim_assessments=(claim_reference(pop_24),),
        evidence_notes=("Future population observations support this empirical loss.",),
    )
    metric = MetricEligibilityAssessment(
        assessment_id="metric-nlpd-24h",
        metric_id="negative_log_predictive_density",
        metric_family=MetricFamily.PREDICTIVE_PROPER_SCORE,
        status=EligibilityStatus.ELIGIBLE,
        scope=pop_24.scope,
        data_source_ids=("processed", "metadata"),
        supporting_claim_assessments=(claim_reference(pop_24),),
        evidence_notes=("Held-out future populations support this proper score.",),
        required_split_unit=ExperimentalUnitLevel.WELL,
        partition_purpose=MetricPartitionPurpose.UNTOUCHED_TEST,
    )
    all_source_ids = tuple(item.source_id for item in (raw, processed, metadata, documentation))
    return DatasetManifest(
        dataset_id="v03-fixture",
        version="2026-08-09",
        title="Manifest v0.3 fixture",
        description="Public-real structural fixture for scoped eligibility tests.",
        origin=PublicRealDataOrigin(
            repository="NCBI GEO",
            study_accession="GSE-V02",
            release="2026-08-09",
            species=(term("Homo sapiens"),),
            biological_systems=(term("cultured cell population"),),
        ),
        sources=(raw, processed, metadata, documentation),
        use_policies=(
            DataUsePolicy(
                policy_id="base-policy",
                source_ids=all_source_ids,
                license_name="CC BY 4.0",
                terms_url="https://creativecommons.org/licenses/by/4.0/",
                reviewed_on=date(2026, 8, 9),
                permissions=permissions(),
            ),
        ),
        slice_spec=DatasetSliceSpec(
            kind=DatasetSliceKind.WHOLE_ARTIFACT,
            slice_id="v03-fixture-whole-artifact",
            selection_source_ids=("processed",),
            record_id_field="cell_id",
            selected_record_ids_sha256="b" * 64,
            selected_record_count=100,
            selected_subject_count=10,
        ),
        experimental_design=design,
        capabilities=capabilities,
        claim_assessments=(pop_24, pop_48, function_24, function_48),
        loss_assessments=(loss,),
        metric_assessments=(metric,),
    )


def revalidate(manifest: DatasetManifest, **updates: object) -> DatasetManifest:
    payload = manifest.model_dump(mode="python")
    payload.update(updates)
    return DatasetManifest.model_validate(payload)


def test_v03_repeats_claims_by_scope_and_repeats_output_terms_by_readout_id() -> None:
    manifest = manifest_factory()
    assert manifest.schema_version == DATASET_MANIFEST_SCHEMA_VERSION == "0.3-experimental"
    assert manifest.claim_assessments[0].claim is manifest.claim_assessments[1].claim
    assert manifest.claim_assessments[0].scope != manifest.claim_assessments[1].scope
    outputs = manifest.capabilities.functional.outputs
    assert outputs[0].output.key == outputs[1].output.key
    assert outputs[0].readout_id != outputs[1].readout_id
    assert DatasetManifest.model_validate_json(manifest.model_dump_json()) == manifest


def test_scope_fingerprint_is_canonical_but_duplicate_semantic_claims_fail() -> None:
    manifest = manifest_factory()
    original = manifest.claim_assessments[0]
    reversed_scope = original.scope.model_copy(
        update={"modalities": tuple(reversed(original.scope.modalities))}
    )
    assert original.scope.fingerprint == AssessmentScope.model_validate(reversed_scope).fingerprint
    duplicate = ClaimAssessment(
        assessment_id="duplicate-semantic-claim",
        claim=original.claim,
        status=original.status,
        identification_basis=original.identification_basis,
        scope=reversed_scope,
        evidence_source_ids=original.evidence_source_ids,
        evidence_notes=original.evidence_notes,
    )
    with pytest.raises(ValidationError, match="claim and canonical assessment scope"):
        revalidate(
            manifest,
            claim_assessments=(*manifest.claim_assessments, duplicate),
        )


def test_signed_zero_cannot_evade_duplicate_scope_detection() -> None:
    manifest = manifest_factory()
    original = manifest.claim_assessments[0]
    equivalent_scope = original.scope.model_copy(update={"inference_cutoff_seconds": -0.0})
    duplicate = original.model_copy(
        update={
            "assessment_id": "duplicate-signed-zero-scope",
            "scope": equivalent_scope,
        }
    )
    assert original.scope.fingerprint == equivalent_scope.fingerprint
    with pytest.raises(ValidationError, match="claim and canonical assessment scope"):
        revalidate(manifest, claim_assessments=(*manifest.claim_assessments, duplicate))


def test_claim_fingerprint_canonicalizes_scope_and_evidence_sets() -> None:
    scope = population_scope(
        horizon=86400.0,
        modalities=(term("transcriptome"), term("surface proteome")),
    )
    reordered_scope = scope.model_copy(update={"modalities": tuple(reversed(scope.modalities))})
    first = ClaimAssessment(
        assessment_id="claim-canonical-fingerprint",
        claim=ScientificClaim.POPULATION_DYNAMICS,
        status=EligibilityStatus.CONDITIONALLY_ELIGIBLE,
        identification_basis=IdentificationBasis.ASSOCIATIONAL,
        scope=scope,
        evidence_source_ids=("processed", "metadata"),
        evidence_notes=("Timing is recorded.", "Population linkage is recorded."),
        assumptions=("Assumption B.", "Assumption A."),
        blockers=("Review B.", "Review A."),
    )
    reordered = ClaimAssessment(
        assessment_id=first.assessment_id,
        claim=first.claim,
        status=first.status,
        identification_basis=first.identification_basis,
        scope=reordered_scope,
        evidence_source_ids=tuple(reversed(first.evidence_source_ids)),
        evidence_notes=tuple(reversed(first.evidence_notes)),
        assumptions=tuple(reversed(first.assumptions)),
        blockers=tuple(reversed(first.blockers)),
    )
    assert first.model_dump(mode="python") != reordered.model_dump(mode="python")
    assert first.fingerprint == reordered.fingerprint

    changed_scope = reordered_scope.model_copy(update={"horizons_seconds": (172800.0,)})
    changed = reordered.model_copy(update={"scope": changed_scope})
    assert changed.fingerprint != first.fingerprint


def test_loss_and_metric_fingerprints_canonicalize_scope_and_set_like_fields() -> None:
    scope = population_scope(
        horizon=86400.0,
        modalities=(term("transcriptome"), term("surface proteome")),
    )
    reordered_scope = scope.model_copy(update={"modalities": tuple(reversed(scope.modalities))})
    references = (
        ClaimAssessmentReference(
            assessment_id="claim-b",
            assessment_fingerprint="b" * 64,
        ),
        ClaimAssessmentReference(
            assessment_id="claim-a",
            assessment_fingerprint="a" * 64,
        ),
    )
    data_source_ids = ("processed", "metadata")
    evidence_notes = ("Evidence B.", "Evidence A.")
    assumptions = ("Assumption B.", "Assumption A.")
    blockers = ("Review B.", "Review A.")
    loss = LossEligibilityAssessment(
        assessment_id="loss-canonical-fingerprint",
        loss_kind=LossKind.MULTI_HORIZON_FUTURE,
        status=EligibilityStatus.CONDITIONALLY_ELIGIBLE,
        scope=scope,
        required_split_unit=ExperimentalUnitLevel.WELL,
        data_source_ids=data_source_ids,
        supporting_claim_assessments=references,
        evidence_notes=evidence_notes,
        assumptions=assumptions,
        blockers=blockers,
    )
    reordered_loss = LossEligibilityAssessment(
        assessment_id=loss.assessment_id,
        loss_kind=loss.loss_kind,
        status=loss.status,
        scope=reordered_scope,
        required_split_unit=loss.required_split_unit,
        data_source_ids=tuple(reversed(data_source_ids)),
        supporting_claim_assessments=tuple(reversed(references)),
        evidence_notes=tuple(reversed(evidence_notes)),
        assumptions=tuple(reversed(assumptions)),
        blockers=tuple(reversed(blockers)),
    )
    assert loss.model_dump(mode="python") != reordered_loss.model_dump(mode="python")
    assert loss.fingerprint == reordered_loss.fingerprint
    assert (
        reordered_loss.model_copy(update={"loss_kind": LossKind.FUNCTIONAL_OUTCOME}).fingerprint
        != loss.fingerprint
    )

    metric = MetricEligibilityAssessment(
        assessment_id="metric-canonical-fingerprint",
        metric_id="negative_log_predictive_density",
        metric_family=MetricFamily.PREDICTIVE_PROPER_SCORE,
        partition_purpose=MetricPartitionPurpose.UNTOUCHED_TEST,
        status=EligibilityStatus.CONDITIONALLY_ELIGIBLE,
        scope=scope,
        required_split_unit=ExperimentalUnitLevel.WELL,
        data_source_ids=data_source_ids,
        supporting_claim_assessments=references,
        evidence_notes=evidence_notes,
        assumptions=assumptions,
        blockers=blockers,
    )
    reordered_metric = MetricEligibilityAssessment(
        assessment_id=metric.assessment_id,
        metric_id=metric.metric_id,
        metric_family=metric.metric_family,
        partition_purpose=metric.partition_purpose,
        status=metric.status,
        scope=reordered_scope,
        required_split_unit=metric.required_split_unit,
        data_source_ids=tuple(reversed(data_source_ids)),
        supporting_claim_assessments=tuple(reversed(references)),
        evidence_notes=tuple(reversed(evidence_notes)),
        assumptions=tuple(reversed(assumptions)),
        blockers=tuple(reversed(blockers)),
    )
    assert metric.model_dump(mode="python") != reordered_metric.model_dump(mode="python")
    assert metric.fingerprint == reordered_metric.fingerprint
    assert (
        reordered_metric.model_copy(
            update={"metric_id": "continuous_ranked_probability_score"}
        ).fingerprint
        != metric.fingerprint
    )


def test_assessment_ids_are_globally_unique() -> None:
    manifest = manifest_factory()
    duplicate_id = manifest.loss_assessments[0].model_copy(
        update={"assessment_id": manifest.claim_assessments[0].assessment_id}
    )
    with pytest.raises(ValidationError, match="dataset assessment IDs"):
        revalidate(manifest, loss_assessments=(duplicate_id,))


def test_objectives_bind_exact_claim_fingerprint_and_scope() -> None:
    manifest = manifest_factory()
    loss = manifest.loss_assessments[0]
    stale_reference = loss.supporting_claim_assessments[0].model_copy(
        update={"assessment_fingerprint": "b" * 64}
    )
    with pytest.raises(ValidationError, match="fingerprint does not match"):
        revalidate(
            manifest,
            loss_assessments=(
                loss.model_copy(update={"supporting_claim_assessments": (stale_reference,)}),
            ),
        )

    wrong_scope = manifest.claim_assessments[1].scope
    with pytest.raises(ValidationError, match="scopes must match exactly"):
        revalidate(
            manifest,
            loss_assessments=(loss.model_copy(update={"scope": wrong_scope}),),
        )


def test_objective_status_cannot_exceed_supporting_claim_status() -> None:
    manifest = manifest_factory()
    original = manifest.claim_assessments[0]
    conditional = ClaimAssessment(
        **{
            **original.model_dump(mode="python"),
            "status": EligibilityStatus.CONDITIONALLY_ELIGIBLE,
            "assumptions": ("The exact population scope is exchangeable.",),
        }
    )
    loss = manifest.loss_assessments[0].model_copy(
        update={"supporting_claim_assessments": (claim_reference(conditional),)}
    )
    with pytest.raises(ValidationError, match="cannot be stronger"):
        revalidate(
            manifest,
            claim_assessments=(conditional,),
            loss_assessments=(loss,),
            metric_assessments=(),
        )


def test_objective_family_and_data_sources_fail_closed() -> None:
    manifest = manifest_factory()
    loss = manifest.loss_assessments[0]
    incompatible = loss.model_copy(update={"loss_kind": LossKind.INTERVENTION_EFFECT})
    with pytest.raises(ValidationError, match="no compatible supporting"):
        revalidate(manifest, loss_assessments=(incompatible,))

    unsupported_model_penalty = loss.model_copy(update={"loss_kind": LossKind.STATE_COMPLEXITY})
    with pytest.raises(ValidationError, match="no compatible supporting"):
        revalidate(manifest, loss_assessments=(unsupported_model_penalty,))

    unrelated_source = loss.model_copy(update={"data_source_ids": ("raw",)})
    with pytest.raises(ValidationError, match="exactly cover all supporting claim evidence"):
        revalidate(manifest, loss_assessments=(unrelated_source,))

    for incomplete_sources in (("metadata",), ("processed",)):
        incomplete = loss.model_copy(update={"data_source_ids": incomplete_sources})
        with pytest.raises(
            ValidationError,
            match="exactly cover all supporting claim evidence",
        ):
            revalidate(manifest, loss_assessments=(incomplete,))

    documentation_only = loss.model_copy(update={"data_source_ids": ("documentation",)})
    with pytest.raises(ValidationError, match="cannot rely only on documentation"):
        revalidate(manifest, loss_assessments=(documentation_only,))


@pytest.mark.parametrize(
    "metric_family",
    (
        MetricFamily.INTERVENTION_RANKING,
        MetricFamily.EVENT_OR_SURVIVAL,
        MetricFamily.OOD_OR_SELECTIVE_RISK,
        MetricFamily.PREDICTIVE_SUFFICIENCY,
        MetricFamily.DECISION_UTILITY,
    ),
)
def test_metrics_without_typed_benchmark_semantics_remain_fail_closed(
    metric_family: MetricFamily,
) -> None:
    manifest = manifest_factory()
    metric = manifest.metric_assessments[0].model_copy(
        update={
            "metric_id": f"deferred-{metric_family.value}",
            "metric_family": metric_family,
        }
    )
    with pytest.raises(ValidationError, match="no compatible supporting"):
        revalidate(manifest, metric_assessments=(metric,))


def test_population_distribution_metric_requires_population_dynamics_scope() -> None:
    manifest = manifest_factory()
    metric = manifest.metric_assessments[0].model_copy(
        update={
            "metric_id": "energy-distance",
            "metric_family": MetricFamily.POPULATION_DISTRIBUTION,
        }
    )
    validated = revalidate(manifest, metric_assessments=(metric,))
    assert validated.metric_assessments[0].metric_family is MetricFamily.POPULATION_DISTRIBUTION

    sampling = SamplingDesign(
        subject_kind=SamplingSubjectKind.INDIVIDUAL_CELL,
        subject_unit=ExperimentalUnitLevel.CELL,
        subject_id_field="cell_id",
        source_ids=("metadata",),
        mode=SamplingMode.LONGITUDINAL_NONDESTRUCTIVE,
        linkage=SubjectLinkage.SAME_CELL,
        time_field="time_seconds",
        source_time_units="s",
    )
    design = manifest.experimental_design.model_copy(
        update={
            "sampling": sampling,
            "randomization_unit": ExperimentalUnitLevel.WELL,
            "matched_control_field": "matched_control_well_id",
        }
    )
    transcriptome = manifest.capabilities.modalities[0].model_copy(
        update={
            "subject_alignment": SubjectAlignment.SAME_CELL,
            "alignment_group": "cell-omics",
            "alignment_key_field": "cell_id",
            "destructive": False,
        }
    )
    intervention_kind = term("small molecule")
    capabilities = manifest.capabilities.model_copy(
        update={
            "modalities": (transcriptome, *manifest.capabilities.modalities[1:]),
            "interventions": InterventionCapability(
                source_ids=("metadata",),
                assignment=AssignmentMechanism.RANDOMIZED,
                kinds=(intervention_kind,),
                start_stop_recorded=True,
                matched_controls_present=True,
            ),
            "timing": manifest.capabilities.timing.model_copy(
                update={
                    "intervention_times_recorded": True,
                    "event_ordering_recorded": True,
                }
            ),
        }
    )
    individual_scope = AssessmentScope(
        subject_kind=SamplingSubjectKind.INDIVIDUAL_CELL,
        system_boundary=SystemBoundary.ISOLATED_CELL,
        biological_systems=(term("cultured cell population"),),
        modalities=(transcriptome.modality,),
        intervention_kinds=(intervention_kind,),
        horizons_seconds=(86400.0,),
        inference_cutoff_seconds=0.0,
    )
    individual_effect = ClaimAssessment(
        assessment_id="claim-individual-intervention-effect",
        claim=ScientificClaim.INTERVENTION_EFFECT,
        status=EligibilityStatus.ELIGIBLE,
        identification_basis=IdentificationBasis.RANDOMIZED_WITHIN_STUDY,
        scope=individual_scope,
        evidence_source_ids=("processed", "metadata"),
        evidence_notes=("Randomized same-cell evidence supports the exact individual scope.",),
    )
    invalid_population_metric = MetricEligibilityAssessment(
        assessment_id="metric-individual-energy-distance",
        metric_id="energy-distance",
        metric_family=MetricFamily.POPULATION_DISTRIBUTION,
        partition_purpose=MetricPartitionPurpose.UNTOUCHED_TEST,
        status=EligibilityStatus.ELIGIBLE,
        scope=individual_scope,
        required_split_unit=ExperimentalUnitLevel.WELL,
        data_source_ids=("processed", "metadata"),
        supporting_claim_assessments=(claim_reference(individual_effect),),
        evidence_notes=("This must not authorize a population-distribution metric.",),
    )
    with pytest.raises(ValidationError, match="population-dynamics claim"):
        revalidate(
            manifest,
            experimental_design=design,
            capabilities=capabilities,
            claim_assessments=(individual_effect,),
            loss_assessments=(),
            metric_assessments=(invalid_population_metric,),
        )


def test_metric_requires_a_structurally_safe_split_unit() -> None:
    manifest = manifest_factory()
    unsafe = manifest.metric_assessments[0].model_copy(
        update={"required_split_unit": ExperimentalUnitLevel.CELL}
    )
    with pytest.raises(ValidationError, match="cannot be finer than a protected unit"):
        revalidate(manifest, metric_assessments=(unsafe,))
    unsafe_loss = manifest.loss_assessments[0].model_copy(
        update={"required_split_unit": ExperimentalUnitLevel.CELL}
    )
    with pytest.raises(ValidationError, match="cannot be finer than a protected unit"):
        revalidate(manifest, loss_assessments=(unsafe_loss,))


def test_exact_functional_readout_prevents_same_output_endpoint_borrowing() -> None:
    manifest = manifest_factory()
    function_24 = manifest.claim_assessments[2]
    function_48 = manifest.claim_assessments[3]
    objective = LossEligibilityAssessment(
        assessment_id="loss-function-wrong-endpoint",
        loss_kind=LossKind.FUNCTIONAL_OUTCOME,
        status=EligibilityStatus.ELIGIBLE,
        scope=function_48.scope,
        required_split_unit=ExperimentalUnitLevel.WELL,
        data_source_ids=("processed", "metadata"),
        supporting_claim_assessments=(claim_reference(function_24),),
        evidence_notes=("The endpoints share a label but not an exact readout identity.",),
    )
    with pytest.raises(ValidationError, match="scopes must match exactly"):
        revalidate(manifest, loss_assessments=(objective,), metric_assessments=())


def test_field_clock_future_claims_remain_fail_closed() -> None:
    manifest = manifest_factory()
    readout = manifest.capabilities.functional.outputs[0].model_copy(
        update={
            "measurement_time_seconds": None,
            "measurement_time_field": "outcome_time",
        }
    )
    capabilities = manifest.capabilities.model_copy(
        update={
            "functional": FunctionalCapability(
                outputs=(readout, manifest.capabilities.functional.outputs[1])
            )
        }
    )
    scope = manifest.claim_assessments[2].scope.model_copy(
        update={
            "inference_cutoff_seconds": None,
            "inference_cutoff_field": "prediction_time",
        }
    )
    field_claim = claim(
        "claim-function-field-clock",
        ScientificClaim.FUNCTIONAL_OUTCOME,
        scope,
        status=EligibilityStatus.CONDITIONALLY_ELIGIBLE,
    )
    with pytest.raises(ValidationError, match=r"fixed cutoff|field-clock"):
        revalidate(
            manifest,
            capabilities=capabilities,
            claim_assessments=(field_claim,),
            loss_assessments=(),
            metric_assessments=(),
        )


def test_all_supported_transported_claims_require_typed_transport_scope() -> None:
    manifest = manifest_factory()
    original = manifest.claim_assessments[0]
    transported = ClaimAssessment(
        **{
            **original.model_dump(mode="python"),
            "status": EligibilityStatus.CONDITIONALLY_ELIGIBLE,
            "identification_basis": IdentificationBasis.TRANSPORTED_UNDER_ASSUMPTIONS,
            "assumptions": ("Source and target populations are exchangeable.",),
        }
    )
    with pytest.raises(ValidationError, match="structured transport scope"):
        revalidate(
            manifest,
            claim_assessments=(transported,),
            loss_assessments=(),
            metric_assessments=(),
        )


def test_counterfactual_claim_can_be_recorded_honestly_as_not_assessed() -> None:
    manifest = manifest_factory()
    not_assessed = ClaimAssessment(
        assessment_id="claim-counterfactual-not-assessed",
        claim=ScientificClaim.COUNTERFACTUAL_GENERALIZATION,
        status=EligibilityStatus.NOT_ASSESSED,
        identification_basis=IdentificationBasis.NONE,
        scope=manifest.claim_assessments[0].scope,
        blockers=("Typed source and target transport scope is not implemented.",),
    )
    validated = revalidate(
        manifest,
        claim_assessments=(not_assessed,),
        loss_assessments=(),
        metric_assessments=(),
    )
    assert validated.claim_assessments == (not_assessed,)


def test_spatial_boundary_cannot_borrow_nonspatial_population_evidence() -> None:
    manifest = manifest_factory()
    original = manifest.claim_assessments[0]
    spatial_scope = original.scope.model_copy(
        update={"system_boundary": SystemBoundary.SPATIAL_TISSUE_NICHE}
    )
    spatial_claim = original.model_copy(update={"scope": spatial_scope})
    with pytest.raises(ValidationError, match="cell-resolved spatial capability"):
        revalidate(
            manifest,
            claim_assessments=(spatial_claim,),
            loss_assessments=(),
            metric_assessments=(),
        )


def test_permitted_use_cannot_hide_an_unresolved_condition() -> None:
    with pytest.raises(ValidationError, match="cannot retain unresolved conditions"):
        UsePermission(
            use_case=DataUseCase.BENCHMARK_EVALUATION,
            status=PermissionStatus.PERMITTED,
            conditions=("Separate legal review is required.",),
        )


def test_prohibited_permission_dominates_scientific_not_assessed_status() -> None:
    manifest = manifest_factory()
    original = manifest.claim_assessments[0]
    not_assessed = original.model_copy(
        update={
            "status": EligibilityStatus.NOT_ASSESSED,
            "identification_basis": IdentificationBasis.NONE,
            "blockers": ("Scientific review has not been performed.",),
        }
    )
    overlay = DataUsePolicy(
        policy_id="prohibited-overlay",
        source_ids=("processed",),
        license_name="Repository terms",
        terms_url="https://example.org/repository-terms",
        reviewed_on=date(2026, 8, 9),
        permissions=permissions(benchmark=PermissionStatus.PROHIBITED),
    )
    layered = revalidate(
        manifest,
        use_policies=(*manifest.use_policies, overlay),
        claim_assessments=(not_assessed,),
        loss_assessments=(),
        metric_assessments=(),
    )
    resolution = layered.resolve_assessment(
        DatasetAssessmentReference(
            dataset_manifest_fingerprint=layered.fingerprint,
            assessment_id=not_assessed.assessment_id,
            assessment_fingerprint=not_assessed.fingerprint,
        ),
        use_case=DataUseCase.BENCHMARK_EVALUATION,
    )
    assert resolution.scientific_status is EligibilityStatus.NOT_ASSESSED
    assert resolution.effective_permission is not None
    assert resolution.effective_permission.status is PermissionStatus.PROHIBITED
    assert resolution.workflow_status is EligibilityStatus.INELIGIBLE
    assert not resolution.use_allowed_without_additional_review


def test_overlapping_policy_layers_and_exact_resolution_keep_science_separate() -> None:
    manifest = manifest_factory()
    overlay = DataUsePolicy(
        policy_id="repository-overlay",
        source_ids=("processed",),
        license_name="Repository terms",
        terms_url="https://example.org/repository-terms",
        reviewed_on=date(2026, 8, 9),
        permissions=permissions(benchmark=PermissionStatus.PROHIBITED),
    )
    layered = revalidate(manifest, use_policies=(*manifest.use_policies, overlay))
    metric = layered.metric_assessments[0]
    reference = DatasetAssessmentReference(
        dataset_manifest_fingerprint=layered.fingerprint,
        assessment_id=metric.assessment_id,
        assessment_fingerprint=metric.fingerprint,
    )
    resolution = layered.resolve_assessment(
        reference,
        use_case=DataUseCase.BENCHMARK_EVALUATION,
    )
    assert resolution.assessment_kind is AssessmentKind.METRIC
    assert resolution.scientific_status is EligibilityStatus.ELIGIBLE
    assert resolution.effective_permission is not None
    assert resolution.effective_permission.status is PermissionStatus.PROHIBITED
    assert resolution.workflow_status is EligibilityStatus.INELIGIBLE
    assert not resolution.use_allowed_without_additional_review
    assert resolution.legal_conditions == ("Benchmark use is prohibited by the repository layer.",)
    assert resolution.effective_permission.applicable_policy_ids == (
        "base-policy",
        "repository-overlay",
    )
    assert resolution.effective_permission.conditions == (
        "Benchmark use is prohibited by the repository layer.",
    )


def test_resolver_rejects_stale_manifest_or_assessment_fingerprints() -> None:
    manifest = manifest_factory()
    metric = manifest.metric_assessments[0]
    exact_reference = DatasetAssessmentReference(
        dataset_manifest_fingerprint=manifest.fingerprint,
        assessment_id=metric.assessment_id,
        assessment_fingerprint=metric.fingerprint,
    )
    permitted = manifest.resolve_assessment(
        exact_reference,
        use_case=DataUseCase.BENCHMARK_EVALUATION,
    )
    assert permitted.workflow_status is EligibilityStatus.ELIGIBLE
    assert permitted.use_allowed_without_additional_review
    forged_resolution = permitted.model_dump(mode="python")
    forged_resolution["data_source_ids"] = ("raw",)
    with pytest.raises(ValidationError, match="sources must match"):
        type(permitted).model_validate(forged_resolution)
    forged_resolution = permitted.model_dump(mode="python")
    forged_resolution["effective_permission"] = None
    with pytest.raises(ValidationError, match="absent exactly when sources are absent"):
        type(permitted).model_validate(forged_resolution)
    with pytest.raises(ValueError, match="does not bind this dataset manifest"):
        manifest.resolve_assessment(
            DatasetAssessmentReference(
                dataset_manifest_fingerprint="b" * 64,
                assessment_id=metric.assessment_id,
                assessment_fingerprint=metric.fingerprint,
            ),
            use_case=DataUseCase.BENCHMARK_EVALUATION,
        )
    with pytest.raises(ValueError, match="fingerprint does not match"):
        manifest.resolve_assessment(
            DatasetAssessmentReference(
                dataset_manifest_fingerprint=manifest.fingerprint,
                assessment_id=metric.assessment_id,
                assessment_fingerprint="b" * 64,
            ),
            use_case=DataUseCase.BENCHMARK_EVALUATION,
        )


def test_v01_payload_is_not_silently_accepted() -> None:
    manifest = manifest_factory()
    payload = manifest.model_dump(mode="python")
    payload["schema_version"] = "0.1-experimental"
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(payload)
