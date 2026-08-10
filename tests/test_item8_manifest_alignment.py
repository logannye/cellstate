from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError
from test_data_manifests import manifest_factory, revalidate, term

from cellstate.data import (
    AssessmentScope,
    ClaimAssessment,
    ClaimAssessmentReference,
    CompositeIdentityEncoding,
    ControlPredicateValueType,
    DatasetAssessmentReference,
    DatasetCapabilities,
    DatasetManifest,
    DataUseCase,
    DataUsePolicy,
    EligibilityStatus,
    ExperimentalDesign,
    ExperimentalUnitLevel,
    ExperimentalUnitSpec,
    FunctionalCapability,
    IdentificationBasis,
    MatchedControlDefinition,
    MatchedControlPredicate,
    MetricEligibilityAssessment,
    MetricFamily,
    MetricPartitionPurpose,
    ModalitySpec,
    PermissionStatus,
    RandomizedEndpointContrast,
    SamplingDesign,
    SamplingMode,
    SamplingSubjectKind,
    ScientificClaim,
    SubjectAlignment,
    SubjectLinkage,
    TimingCapability,
    UnitIdentityExpression,
    UnitIdentityExpressionKind,
    UsePermission,
)
from cellstate.domain import SystemBoundary


def source_field(name: str) -> UnitIdentityExpression:
    return UnitIdentityExpression(
        kind=UnitIdentityExpressionKind.SOURCE_FIELD,
        source_fields=(name,),
    )


def composite(*names: str) -> UnitIdentityExpression:
    return UnitIdentityExpression(
        kind=UnitIdentityExpressionKind.COMPOSITE_SOURCE_FIELDS,
        source_fields=names,
        composite_encoding=CompositeIdentityEncoding.CANONICAL_JSON_UTF8_STRING_ARRAY_V1,
    )


def endpoint_manifest() -> DatasetManifest:
    manifest = manifest_factory()
    plate_identity = source_field("plate_id")
    well_identity = composite("plate_id", "well_id")
    units = (
        ExperimentalUnitSpec(
            level=ExperimentalUnitLevel.PLATE,
            identity=plate_identity,
            source_ids=("metadata",),
        ),
        ExperimentalUnitSpec(
            level=ExperimentalUnitLevel.WELL,
            identity=well_identity,
            source_ids=("metadata",),
            parent_level=ExperimentalUnitLevel.PLATE,
        ),
        ExperimentalUnitSpec(
            level=ExperimentalUnitLevel.CELL,
            identity=source_field("cell_id"),
            source_ids=("processed",),
            parent_level=ExperimentalUnitLevel.WELL,
        ),
    )
    matched_control = MatchedControlDefinition(
        predicates=(
            MatchedControlPredicate(
                source_field="dose_value",
                value_type=ControlPredicateValueType.NUMBER,
                equals=0.0,
            ),
            MatchedControlPredicate(
                source_field="perturbation",
                value_type=ControlPredicateValueType.STRING,
                equals="Vehicle",
            ),
        ),
        stratum_identity=plate_identity,
        source_ids=("metadata",),
    )
    design = ExperimentalDesign(
        units=units,
        sampling=SamplingDesign(
            subject_kind=SamplingSubjectKind.POPULATION,
            subject_unit=ExperimentalUnitLevel.WELL,
            subject_identity=well_identity,
            source_ids=("metadata",),
            mode=SamplingMode.ENDPOINT_DESTRUCTIVE,
            linkage=SubjectLinkage.NONE,
        ),
        default_split_unit=ExperimentalUnitLevel.WELL,
        biological_replicate_unit=ExperimentalUnitLevel.WELL,
        randomization_unit=ExperimentalUnitLevel.WELL,
        randomized_endpoint_contrast=RandomizedEndpointContrast(
            assignment_time_seconds=0.0,
            endpoint_time_seconds=86400.0,
            baseline_observation_present=False,
            matched_control=matched_control,
            source_ids=("metadata",),
        ),
        batch_fields=("plate_id",),
    )
    transcriptome = term("transcriptome")
    modality = ModalitySpec(
        modality=transcriptome,
        source_ids=("raw", "processed"),
        subject_alignment=SubjectAlignment.SAME_POPULATION,
        alignment_group="well-transcriptome",
        alignment_unit=ExperimentalUnitLevel.WELL,
        raw_available=True,
        processed_available=True,
        destructive=True,
        feature_identifier_namespace="Ensembl",
    )
    capabilities = manifest.capabilities.model_copy(
        update={
            "modalities": (modality,),
            "timing": TimingCapability(
                source_ids=("metadata",),
                timepoints_seconds=(86400.0,),
                observation_times_recorded=True,
                intervention_times_recorded=True,
                event_ordering_recorded=True,
            ),
            "functional": FunctionalCapability(),
            "environment": manifest.capabilities.environment.model_copy(update={"variables": ()}),
        }
    )
    claim_scope = AssessmentScope(
        subject_kind=SamplingSubjectKind.POPULATION,
        system_boundary=SystemBoundary.POPULATION,
        biological_systems=(term("cultured cell population"),),
        modalities=(transcriptome,),
        intervention_kinds=(term("small molecule exposure"),),
        horizons_seconds=(86400.0,),
        inference_cutoff_seconds=0.0,
    )
    claim = ClaimAssessment(
        assessment_id="randomized-k562-endpoint-effect",
        claim=ScientificClaim.INTERVENTION_EFFECT,
        status=EligibilityStatus.ELIGIBLE,
        identification_basis=IdentificationBasis.RANDOMIZED_WITHIN_STUDY,
        scope=claim_scope,
        evidence_source_ids=("processed", "metadata", "documentation"),
        execution_source_ids=("processed", "metadata"),
        evidence_notes=(
            "Randomized assignment metadata and the assay endpoint support this exact contrast.",
        ),
    )
    return revalidate(
        manifest,
        experimental_design=design,
        capabilities=DatasetCapabilities.model_validate(capabilities),
        claim_assessments=(claim,),
    )


def test_typed_unit_identities_cover_field_composite_and_constant() -> None:
    field_identity = source_field("well_id")
    composite_identity = composite("plate", "well")
    assert field_identity.kind is UnitIdentityExpressionKind.SOURCE_FIELD
    assert field_identity.evaluate({"well_id": "A:01"}) == "A:01"
    assert composite_identity.composite_encoding is not None
    assert composite_identity.evaluate({"plate": "rep1:plate1", "well": "A:01"}) == (
        '["rep1:plate1","A:01"]'
    )
    constant = UnitIdentityExpression(
        kind=UnitIdentityExpressionKind.MANIFEST_CONSTANT,
        constant_value="K562",
    )
    assert constant.constant_value == "K562"
    assert constant.evaluate({}) == "K562"

    with pytest.raises(ValueError, match="source field is absent"):
        field_identity.evaluate({})
    with pytest.raises(ValueError, match="exact strings"):
        field_identity.evaluate({"well_id": 1})

    with pytest.raises(ValidationError, match="at least two fields"):
        UnitIdentityExpression(
            kind=UnitIdentityExpressionKind.COMPOSITE_SOURCE_FIELDS,
            source_fields=("well",),
            composite_encoding=(CompositeIdentityEncoding.CANONICAL_JSON_UTF8_STRING_ARRAY_V1),
        )
    with pytest.raises(ValidationError, match="exactly one legacy ID field or typed identity"):
        ExperimentalUnitSpec(
            level=ExperimentalUnitLevel.WELL,
            id_field="well_id",
            identity=source_field("well_id"),
            source_ids=("metadata",),
        )


def test_sampling_and_modality_alignment_fail_closed_on_near_match() -> None:
    manifest = endpoint_manifest()
    assert manifest.experimental_design.sampling.subject_id_field is None
    assert manifest.capabilities.modalities[0].alignment_unit is ExperimentalUnitLevel.WELL

    wrong_sampling = manifest.experimental_design.sampling.model_copy(
        update={
            "subject_identity": composite("well_id", "plate_id"),
        }
    )
    with pytest.raises(ValidationError, match="subject identity must match"):
        revalidate(
            manifest,
            experimental_design=manifest.experimental_design.model_copy(
                update={"sampling": wrong_sampling}
            ),
        )

    wrong_alignment = manifest.capabilities.modalities[0].model_copy(
        update={
            "alignment_unit": None,
            "alignment_identity": composite("well_id", "plate_id"),
        }
    )
    with pytest.raises(ValidationError, match="alignment key must match"):
        revalidate(
            manifest,
            capabilities=manifest.capabilities.model_copy(
                update={"modalities": (wrong_alignment,)}
            ),
        )


def test_endpoint_contrast_needs_no_fabricated_baseline_observation() -> None:
    manifest = endpoint_manifest()
    contrast = manifest.experimental_design.randomized_endpoint_contrast
    assert contrast is not None
    assert contrast.assignment_time_seconds == 0.0
    assert contrast.baseline_observation_present is False
    assert manifest.capabilities.timing.timepoints_seconds == (86400.0,)

    control = contrast.matched_control
    assert control.matches({"dose_value": 0.0, "perturbation": "Vehicle"})
    assert not control.matches({"dose_value": 100.0, "perturbation": "Vehicle"})
    assert not control.matches({"dose_value": 0, "perturbation": "Vehicle"})

    with pytest.raises(ValidationError, match="at least 2 items"):
        MatchedControlDefinition(
            predicates=(control.predicates[1],),
            stratum_identity=control.stratum_identity,
            source_ids=control.source_ids,
        )
    with pytest.raises(ValidationError, match="no baseline observation"):
        RandomizedEndpointContrast(
            assignment_time_seconds=0.0,
            endpoint_time_seconds=86400.0,
            baseline_observation_present=True,  # type: ignore[arg-type]
            matched_control=control,
            source_ids=("metadata",),
        )

    no_exact_contrast = manifest.experimental_design.model_copy(
        update={"randomized_endpoint_contrast": None}
    )
    with pytest.raises(ValidationError, match="timed exposure"):
        revalidate(manifest, experimental_design=no_exact_contrast)


def test_review_sources_do_not_expand_exact_execution_permission_scope() -> None:
    manifest = endpoint_manifest()
    original_policy = manifest.use_policies[0]

    def permissions(status: PermissionStatus) -> tuple[UsePermission, ...]:
        return tuple(
            UsePermission(
                use_case=use_case,
                status=status,
                conditions=("Documentation is review-only.",)
                if status is PermissionStatus.PROHIBITED
                else (),
            )
            for use_case in DataUseCase
        )

    policies = (
        original_policy.model_copy(update={"source_ids": ("raw", "processed", "metadata")}),
        DataUsePolicy(
            policy_id="review-documentation-only",
            source_ids=("documentation",),
            license_name="review-only fixture",
            terms_url="https://example.org/review-terms",
            reviewed_on=date(2026, 8, 9),
            permissions=permissions(PermissionStatus.PROHIBITED),
        ),
    )
    manifest = revalidate(manifest, use_policies=policies)
    claim = manifest.claim_assessments[0]
    resolution = manifest.resolve_assessment(
        DatasetAssessmentReference(
            dataset_manifest_fingerprint=manifest.fingerprint,
            assessment_id=claim.assessment_id,
            assessment_fingerprint=claim.fingerprint,
        ),
        use_case=DataUseCase.BENCHMARK_EVALUATION,
    )
    assert resolution.data_source_ids == ("metadata", "processed")
    assert resolution.effective_permission is not None
    assert resolution.effective_permission.status is PermissionStatus.PERMITTED

    legacy_claim = claim.model_copy(update={"execution_source_ids": None})
    legacy_manifest = revalidate(manifest, claim_assessments=(legacy_claim,))
    legacy_resolution = legacy_manifest.resolve_assessment(
        DatasetAssessmentReference(
            dataset_manifest_fingerprint=legacy_manifest.fingerprint,
            assessment_id=legacy_claim.assessment_id,
            assessment_fingerprint=legacy_claim.fingerprint,
        ),
        use_case=DataUseCase.BENCHMARK_EVALUATION,
    )
    assert legacy_resolution.effective_permission is not None
    assert legacy_resolution.effective_permission.status is PermissionStatus.PROHIBITED

    documentation_execution = claim.model_copy(update={"execution_source_ids": ("documentation",)})
    with pytest.raises(ValidationError, match="execution sources cannot be documentation"):
        revalidate(manifest, claim_assessments=(documentation_execution,))

    documentation_only = claim.model_copy(
        update={
            "evidence_source_ids": ("documentation",),
            "execution_source_ids": ("documentation",),
        }
    )
    with pytest.raises(ValidationError, match="cannot rely only on documentation"):
        revalidate(manifest, claim_assessments=(documentation_only,))


def test_population_distribution_metric_accepts_population_intervention_effect() -> None:
    manifest = endpoint_manifest()
    claim = manifest.claim_assessments[0]
    metric = MetricEligibilityAssessment(
        assessment_id="population-energy-score-test",
        status=EligibilityStatus.ELIGIBLE,
        scope=claim.scope,
        required_split_unit=ExperimentalUnitLevel.WELL,
        data_source_ids=("processed", "metadata"),
        supporting_claim_assessments=(
            ClaimAssessmentReference(
                assessment_id=claim.assessment_id,
                assessment_fingerprint=claim.fingerprint,
            ),
        ),
        evidence_notes=("Equal-weight well distributions are directly scored.",),
        metric_id="energy_score.population",
        metric_family=MetricFamily.POPULATION_DISTRIBUTION,
        partition_purpose=MetricPartitionPurpose.UNTOUCHED_TEST,
    )
    admitted = revalidate(manifest, metric_assessments=(metric,))
    assert admitted.metric_assessments == (metric,)

    leaked_review_source = metric.model_copy(
        update={"data_source_ids": ("processed", "metadata", "documentation")}
    )
    with pytest.raises(ValidationError, match="execution data"):
        revalidate(manifest, metric_assessments=(leaked_review_source,))
