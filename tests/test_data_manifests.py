from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from cellstate.data import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    AssignmentMechanism,
    ClaimEligibility,
    ClaimScope,
    ClaimStatus,
    DatasetCapabilities,
    DatasetManifest,
    DataUseCase,
    DataUsePolicy,
    EnvironmentCapability,
    EnvironmentVariable,
    ExperimentalDesign,
    ExperimentalUnitLevel,
    ExperimentalUnitSpec,
    FunctionalCapability,
    FunctionalReadout,
    IdentificationBasis,
    InterventionCapability,
    LineageCapability,
    LineageResolution,
    ModalitySpec,
    PermissionStatus,
    PublicRealDataOrigin,
    ReadoutStatus,
    RealizationEvidence,
    SamplingDesign,
    SamplingMode,
    SamplingSubjectKind,
    ScientificClaim,
    SourceArtifact,
    SourceKind,
    SpatialCapability,
    SpatialResolution,
    SubjectAlignment,
    SubjectLinkage,
    TimingCapability,
    UsePermission,
)
from cellstate.domain import OntologyTerm, SystemBoundary


def term(label: str) -> OntologyTerm:
    return OntologyTerm(label=label)


def source(source_id: str, kind: SourceKind) -> SourceArtifact:
    suffix = "h5ad" if kind in {SourceKind.RAW, SourceKind.PROCESSED} else "json"
    return SourceArtifact(
        source_id=source_id,
        kind=kind,
        uri=f"https://example.org/data/{source_id}.{suffix}",
        sha256="A" * 64,
        media_type="application/octet-stream",
        accession="GSE000001",
        release="2024-01-01",
        byte_count=1024,
        retrieved_at=datetime(2026, 8, 9, tzinfo=UTC),
    )


def permissions(
    *,
    commercial: PermissionStatus = PermissionStatus.CONDITIONAL,
) -> tuple[UsePermission, ...]:
    return tuple(
        UsePermission(
            use_case=use_case,
            status=(
                commercial
                if use_case is DataUseCase.COMMERCIAL_MODEL_TRAINING
                else PermissionStatus.PERMITTED
            ),
            conditions=("Separate permission review is required.",)
            if use_case is DataUseCase.COMMERCIAL_MODEL_TRAINING
            else (),
        )
        for use_case in DataUseCase
    )


def scope(
    *,
    subject_kind: SamplingSubjectKind = SamplingSubjectKind.POPULATION,
    system_boundary: SystemBoundary = SystemBoundary.POPULATION,
    modalities: tuple[OntologyTerm, ...] = (),
    interventions: tuple[OntologyTerm, ...] = (),
    outputs: tuple[OntologyTerm, ...] = (),
    environments: tuple[OntologyTerm, ...] = (),
    horizons: tuple[float, ...] = (),
) -> ClaimScope:
    return ClaimScope(
        subject_kind=subject_kind,
        system_boundary=system_boundary,
        biological_systems=(term("cultured cell population"),),
        modalities=modalities,
        intervention_kinds=interventions,
        functional_outputs=outputs,
        environment_variables=environments,
        horizons_seconds=horizons,
        inference_cutoff_seconds=0.0 if horizons else None,
    )


def eligible(
    claim: ScientificClaim,
    claim_scope: ClaimScope,
    *,
    basis: IdentificationBasis = IdentificationBasis.DESCRIPTIVE,
    status: ClaimStatus = ClaimStatus.ELIGIBLE,
) -> ClaimEligibility:
    return ClaimEligibility(
        claim=claim,
        status=status,
        identification_basis=basis,
        scope=claim_scope,
        evidence_source_ids=("processed", "metadata"),
        evidence_notes=("The declared experimental structure supports this scoped role.",),
        assumptions=("Exchangeability holds within the declared scope.",)
        if status is ClaimStatus.CONDITIONALLY_ELIGIBLE
        else (),
    )


def manifest_factory() -> DatasetManifest:
    raw = source("raw", SourceKind.RAW)
    processed = source("processed", SourceKind.PROCESSED)
    metadata = source("metadata", SourceKind.METADATA)
    documentation = source("documentation", SourceKind.DOCUMENTATION)
    design = ExperimentalDesign(
        units=(
            ExperimentalUnitSpec(
                level=ExperimentalUnitLevel.PLATE,
                id_field="plate_id",
                source_ids=("metadata",),
            ),
            ExperimentalUnitSpec(
                level=ExperimentalUnitLevel.WELL,
                id_field="well_id",
                source_ids=("metadata",),
                parent_level=ExperimentalUnitLevel.PLATE,
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
            time_field="hours_after_treatment",
            source_time_units="h",
        ),
        default_split_unit=ExperimentalUnitLevel.WELL,
        biological_replicate_unit=ExperimentalUnitLevel.WELL,
        randomization_unit=ExperimentalUnitLevel.WELL,
        matched_control_field="control_well_id",
        batch_fields=("plate_id", "library_batch"),
    )
    drug = term("small molecule exposure")
    viability = term("cell viability")
    culture_medium = term("culture medium")
    transcriptome = term("transcriptome")
    surface_proteome = term("surface proteome")
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
                feature_identifier_namespace="Ensembl",
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
        interventions=InterventionCapability(
            source_ids=("metadata",),
            assignment=AssignmentMechanism.RANDOMIZED,
            kinds=(drug,),
            targets_recorded=True,
            doses_recorded=True,
            durations_recorded=True,
            start_stop_recorded=True,
            assignment_probabilities_recorded=True,
            matched_controls_present=True,
            realization_evidence=RealizationEvidence.ASSIGNMENT_ONLY,
        ),
        timing=TimingCapability(
            source_ids=("metadata",),
            timepoints_seconds=(0.0, 86400.0),
            observation_times_recorded=True,
            intervention_times_recorded=True,
            environment_times_recorded=True,
            event_ordering_recorded=True,
        ),
        functional=FunctionalCapability(
            outputs=(
                FunctionalReadout(
                    output=viability,
                    source_ids=("processed",),
                    units="fraction",
                    aggregation_level=ExperimentalUnitLevel.WELL,
                    subject_alignment=SubjectAlignment.SAME_POPULATION,
                    alignment_group="well-outcome",
                    alignment_key_field="well_id",
                    status=ReadoutStatus.DIRECT,
                    measurement_time_seconds=86400.0,
                ),
            )
        ),
        environment=EnvironmentCapability(
            variables=(
                EnvironmentVariable(
                    variable=culture_medium,
                    source_ids=("metadata",),
                    measured=False,
                    assigned=True,
                    time_resolved=True,
                ),
            )
        ),
    )
    sample_fusion_scope = scope(modalities=(transcriptome, surface_proteome))
    intervention_scope = scope(
        modalities=(transcriptome,),
        interventions=(drug,),
        outputs=(viability,),
        environments=(culture_medium,),
        horizons=(86400.0,),
    )
    all_sources = tuple(item.source_id for item in (raw, processed, metadata, documentation))
    return DatasetManifest(
        dataset_id="public-study-1",
        version="2024-01-01",
        title="Public perturbation study",
        description="A typed manifest for a real public experimental dataset.",
        origin=PublicRealDataOrigin(
            repository="NCBI GEO",
            study_accession="GSE000001",
            publication_doi="10.0000/example",
            release="2024-01-01",
            species=(term("Homo sapiens"),),
            biological_systems=(term("cultured cell population"),),
        ),
        sources=(raw, processed, metadata, documentation),
        use_policies=(
            DataUsePolicy(
                policy_id="study-policy",
                source_ids=all_sources,
                license_name="CC BY 4.0",
                terms_url="https://creativecommons.org/licenses/by/4.0/",
                reviewed_on=date(2026, 8, 9),
                spdx_identifier="CC-BY-4.0",
                permissions=permissions(),
                attribution_requirements=("Cite the source publication.",),
            ),
        ),
        experimental_design=design,
        capabilities=capabilities,
        claim_eligibility=(
            eligible(ScientificClaim.SAMPLE_LEVEL_MULTIMODAL_FUSION, sample_fusion_scope),
            eligible(
                ScientificClaim.INTERVENTION_EFFECT,
                intervention_scope,
                basis=IdentificationBasis.RANDOMIZED_WITHIN_STUDY,
            ),
            eligible(
                ScientificClaim.FUNCTIONAL_OUTCOME,
                intervention_scope,
                basis=IdentificationBasis.RANDOMIZED_WITHIN_STUDY,
            ),
        ),
    )


def revalidate(manifest: DatasetManifest, **updates: object) -> DatasetManifest:
    payload = manifest.model_dump(mode="python")
    payload.update(updates)
    return DatasetManifest.model_validate(payload)


def test_manifest_round_trip_schema_fingerprint_and_strictness() -> None:
    manifest = manifest_factory()
    assert manifest.schema_version == DATASET_MANIFEST_SCHEMA_VERSION
    assert DatasetManifest.model_validate_json(manifest.model_dump_json()) == manifest
    assert revalidate(manifest).fingerprint == manifest.fingerprint
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate({**manifest.model_dump(mode="python"), "version": 1})
    with pytest.raises(ValidationError):
        PublicRealDataOrigin(
            repository="GEO",
            study_accession="GSE1",
            release="1",
            publicly_downloadable=1,  # type: ignore[arg-type]
            species=(term("human"),),
            biological_systems=(term("cell"),),
        )


def test_source_requires_remote_canonical_content_and_aware_retrieval_time() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        SourceArtifact(
            source_id="bad",
            kind=SourceKind.RAW,
            uri="https://example.org/data",
            sha256="a" * 64,
            media_type="x/data",
            accession="GSE1",
            release="1",
            byte_count=1,
            retrieved_at=datetime(2026, 1, 1),
        )
    with pytest.raises(ValidationError, match=r"credentials|localhost"):
        SourceArtifact(
            source_id="bad",
            kind=SourceKind.RAW,
            uri="https://user:secret@localhost/data",
            sha256="a" * 64,
            media_type="x/data",
            accession="GSE1",
            release="1",
            byte_count=1,
            retrieved_at=datetime.now(UTC),
        )
    assert source("raw", SourceKind.RAW).sha256 == "a" * 64


def test_every_scientific_source_must_link_to_the_declared_origin() -> None:
    manifest = manifest_factory()
    decoy_sources = tuple(
        item
        if item.kind is SourceKind.DOCUMENTATION
        else item.model_copy(update={"accession": f"SUPP-{item.source_id}", "release": "v2"})
        for item in manifest.sources
    )
    with pytest.raises(ValidationError, match="every raw, processed, or metadata source"):
        revalidate(manifest, sources=decoy_sources)

    linked_supplements = tuple(
        item
        if item.kind is SourceKind.DOCUMENTATION
        else item.model_copy(
            update={
                "parent_study_accession": manifest.origin.study_accession,
                "parent_study_release": manifest.origin.release,
            }
        )
        for item in decoy_sources
    )
    assert revalidate(manifest, sources=linked_supplements).sources == linked_supplements

    incomplete_link = source("raw", SourceKind.RAW).model_copy(
        update={"parent_study_accession": "GSE000001"}
    )
    with pytest.raises(ValidationError, match="accession and release together"):
        SourceArtifact.model_validate(incomplete_link)

    contradictory = manifest.sources[0].model_copy(
        update={"parent_study_accession": "GSE-DECOY", "parent_study_release": "v1"}
    )
    with pytest.raises(ValidationError, match="every raw, processed, or metadata source"):
        revalidate(manifest, sources=(contradictory, *manifest.sources[1:]))


def test_use_policy_is_complete_source_scoped_and_executable() -> None:
    manifest = manifest_factory()
    assert (
        manifest.permission_status(DataUseCase.RESEARCH_MODEL_TRAINING)
        is PermissionStatus.PERMITTED
    )
    assert (
        manifest.permission_status(DataUseCase.COMMERCIAL_MODEL_TRAINING)
        is PermissionStatus.CONDITIONAL
    )
    with pytest.raises(ValueError, match="unknown source IDs"):
        manifest.permission_status(DataUseCase.BENCHMARK_EVALUATION, source_ids=("missing",))
    with pytest.raises(ValueError, match="must not be empty"):
        manifest.permission_status(DataUseCase.BENCHMARK_EVALUATION, source_ids=())
    with pytest.raises(ValidationError, match="nonempty after trimming"):
        UsePermission(
            use_case=DataUseCase.COMMERCIAL_MODEL_TRAINING,
            status=PermissionStatus.CONDITIONAL,
            conditions=(" ",),
        )
    with pytest.raises(ValidationError, match="exactly one permission"):
        DataUsePolicy(
            policy_id="bad",
            source_ids=("raw",),
            license_name="custom",
            terms_url="https://example.org/terms",
            reviewed_on=date(2026, 8, 9),
            permissions=(
                UsePermission(
                    use_case=DataUseCase.RESEARCH_MODEL_TRAINING,
                    status=PermissionStatus.PERMITTED,
                ),
            ),
        )
    incomplete_policy = manifest.use_policies[0].model_copy(
        update={"source_ids": ("raw", "processed", "metadata")}
    )
    with pytest.raises(ValidationError, match="cover every source exactly once"):
        revalidate(manifest, use_policies=(incomplete_policy,))


def test_sampling_subject_binds_to_unit_and_repeated_mode() -> None:
    manifest = manifest_factory()
    wrong_field = manifest.experimental_design.sampling.model_copy(
        update={"subject_id_field": "cell_id"}
    )
    with pytest.raises(ValidationError, match="must match its declared unit"):
        revalidate(
            manifest,
            experimental_design=manifest.experimental_design.model_copy(
                update={"sampling": wrong_field}
            ),
        )
    with pytest.raises(ValidationError, match="population linkage"):
        SamplingDesign(
            subject_kind=SamplingSubjectKind.POPULATION,
            subject_unit=ExperimentalUnitLevel.SAMPLE,
            subject_id_field="sample_id",
            source_ids=("metadata",),
            mode=SamplingMode.REPEATED_POPULATION_DESTRUCTIVE,
            linkage=SubjectLinkage.NONE,
            time_field="time",
            source_time_units="h",
        )


def test_split_and_biological_replicate_cannot_be_finer_than_randomization() -> None:
    manifest = manifest_factory()
    unsafe_split = manifest.experimental_design.model_copy(
        update={"default_split_unit": ExperimentalUnitLevel.CELL}
    )
    with pytest.raises(ValidationError, match="default split unit cannot be finer"):
        revalidate(manifest, experimental_design=unsafe_split)
    unsafe_replicate = manifest.experimental_design.model_copy(
        update={"biological_replicate_unit": ExperimentalUnitLevel.CELL}
    )
    with pytest.raises(ValidationError, match="biological replicate unit cannot be finer"):
        revalidate(manifest, experimental_design=unsafe_replicate)


def test_multimodal_claims_distinguish_same_cell_from_same_sample() -> None:
    manifest = manifest_factory()
    same_cell_claim = eligible(
        ScientificClaim.SAME_CELL_MULTIMODAL_FUSION,
        scope(modalities=(term("transcriptome"), term("surface proteome"))),
    )
    with pytest.raises(ValidationError, match="correctly aligned modalities"):
        revalidate(manifest, claim_eligibility=(same_cell_claim,))
    mismatched = manifest.capabilities.modalities[1].model_copy(
        update={"alignment_key_field": "different_sample_id"}
    )
    capabilities = manifest.capabilities.model_copy(
        update={"modalities": (manifest.capabilities.modalities[0], mismatched)}
    )
    with pytest.raises(ValidationError, match="alignment key must match"):
        revalidate(
            manifest,
            capabilities=capabilities,
            claim_eligibility=(manifest.claim_eligibility[0],),
        )


def test_claim_scope_references_declared_capabilities() -> None:
    manifest = manifest_factory()
    bad = eligible(
        ScientificClaim.SNAPSHOT_STATE_PRIOR,
        scope(modalities=(term("metabolome"),)),
    )
    with pytest.raises(ValidationError, match="unsupported modality"):
        revalidate(manifest, claim_eligibility=(bad,))
    with pytest.raises(ValidationError, match="exactly one inference cutoff"):
        ClaimScope(
            subject_kind=SamplingSubjectKind.POPULATION,
            system_boundary=SystemBoundary.POPULATION,
            biological_systems=(term("cell"),),
            horizons_seconds=(10.0,),
        )


def test_claim_explanations_reject_whitespace_only_entries() -> None:
    base = eligible(
        ScientificClaim.SNAPSHOT_STATE_PRIOR,
        scope(modalities=(term("transcriptome"),)),
    )
    for field in ("evidence_notes", "assumptions", "blockers"):
        payload = base.model_dump(mode="python")
        payload[field] = (" \t",)
        with pytest.raises(ValidationError, match="nonempty after trimming"):
            ClaimEligibility.model_validate(payload)


def test_conditional_claims_receive_the_same_capability_gates() -> None:
    manifest = manifest_factory()
    conditional = eligible(
        ScientificClaim.POPULATION_DYNAMICS,
        scope(horizons=(3600.0,)),
        basis=IdentificationBasis.ASSOCIATIONAL,
        status=ClaimStatus.CONDITIONALLY_ELIGIBLE,
    )
    with pytest.raises(ValidationError, match="linked populations"):
        revalidate(manifest, claim_eligibility=(conditional,))


def test_population_dynamics_rejects_unlinked_endpoint_sampling() -> None:
    manifest = manifest_factory()
    endpoint = manifest.experimental_design.sampling.model_copy(
        update={
            "mode": SamplingMode.ENDPOINT_DESTRUCTIVE,
            "linkage": SubjectLinkage.NONE,
            "time_field": None,
            "source_time_units": None,
        }
    )
    design = manifest.experimental_design.model_copy(update={"sampling": endpoint})
    population_claim = eligible(
        ScientificClaim.POPULATION_DYNAMICS,
        scope(modalities=(term("transcriptome"),), horizons=(86400.0,)),
        basis=IdentificationBasis.ASSOCIATIONAL,
    )
    with pytest.raises(ValidationError, match="repeated linked populations"):
        revalidate(manifest, experimental_design=design, claim_eligibility=(population_claim,))


def test_individual_dynamics_requires_nondestructive_same_cell_modality() -> None:
    manifest = manifest_factory()
    sampling = SamplingDesign(
        subject_kind=SamplingSubjectKind.INDIVIDUAL_CELL,
        subject_unit=ExperimentalUnitLevel.CELL,
        subject_id_field="cell_id",
        source_ids=("processed",),
        mode=SamplingMode.LONGITUDINAL_NONDESTRUCTIVE,
        linkage=SubjectLinkage.SAME_CELL,
        time_field="time",
        source_time_units="s",
    )
    design = manifest.experimental_design.model_copy(update={"sampling": sampling})
    claim = eligible(
        ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS,
        scope(
            subject_kind=SamplingSubjectKind.INDIVIDUAL_CELL,
            system_boundary=SystemBoundary.ISOLATED_CELL,
            modalities=(term("transcriptome"),),
            horizons=(3600.0,),
        ),
        basis=IdentificationBasis.ASSOCIATIONAL,
    )
    with pytest.raises(ValidationError, match="nondestructive same-cell evidence"):
        revalidate(manifest, experimental_design=design, claim_eligibility=(claim,))


def test_lineage_fate_requires_future_fate_output() -> None:
    manifest = manifest_factory()
    sampling = SamplingDesign(
        subject_kind=SamplingSubjectKind.CLONE,
        subject_unit=ExperimentalUnitLevel.CLONE,
        subject_id_field="clone_id",
        source_ids=("metadata",),
        mode=SamplingMode.LINEAGE_LINKED_ENDPOINT,
        linkage=SubjectLinkage.SAME_CLONE,
        time_field="time",
        source_time_units="h",
    )
    clone_unit = ExperimentalUnitSpec(
        level=ExperimentalUnitLevel.CLONE,
        id_field="clone_id",
        source_ids=("metadata",),
        parent_level=ExperimentalUnitLevel.SAMPLE,
    )
    design = manifest.experimental_design.model_copy(
        update={"units": (*manifest.experimental_design.units, clone_unit), "sampling": sampling}
    )
    capabilities = manifest.capabilities.model_copy(
        update={
            "lineage": LineageCapability(
                resolution=LineageResolution.CLONE,
                source_ids=("metadata",),
                lineage_ids_recorded=True,
            )
        }
    )
    claim = eligible(
        ScientificClaim.LINEAGE_FATE,
        scope(
            subject_kind=SamplingSubjectKind.CLONE,
            system_boundary=SystemBoundary.CLONE,
            horizons=(86400.0,),
        ),
        basis=IdentificationBasis.ASSOCIATIONAL,
    )
    with pytest.raises(ValidationError, match="future time, and fate output"):
        revalidate(
            manifest,
            experimental_design=design,
            capabilities=capabilities,
            claim_eligibility=(claim,),
        )


def test_future_function_requires_comparable_time_and_horizon_coverage() -> None:
    manifest = manifest_factory()
    too_early = manifest.capabilities.functional.outputs[0].model_copy(
        update={"measurement_time_seconds": 3600.0}
    )
    capabilities = manifest.capabilities.model_copy(
        update={"functional": FunctionalCapability(outputs=(too_early,))}
    )
    with pytest.raises(ValidationError, match="does not cover the claim horizon"):
        revalidate(manifest, capabilities=capabilities)
    derived = manifest.capabilities.functional.outputs[0].model_copy(
        update={"status": ReadoutStatus.DERIVED}
    )
    capabilities = manifest.capabilities.model_copy(
        update={"functional": FunctionalCapability(outputs=(derived,))}
    )
    with pytest.raises(ValidationError, match="conditionally eligible"):
        revalidate(manifest, capabilities=capabilities)

    beyond_timing = manifest.capabilities.functional.outputs[0].model_copy(
        update={"measurement_time_seconds": 172800.0}
    )
    capabilities = manifest.capabilities.model_copy(
        update={"functional": FunctionalCapability(outputs=(beyond_timing,))}
    )
    with pytest.raises(ValidationError, match="outside declared temporal support"):
        revalidate(manifest, capabilities=capabilities)


def test_intervention_and_selection_claims_require_scoped_randomized_evidence() -> None:
    manifest = manifest_factory()
    no_intervention = manifest.capabilities.model_copy(
        update={"interventions": InterventionCapability()}
    )
    selection = eligible(
        ScientificClaim.RETROSPECTIVE_INTERVENTION_SELECTION,
        scope(
            modalities=(term("transcriptome"),),
            outputs=(term("cell viability"),),
            horizons=(86400.0,),
        ),
        basis=IdentificationBasis.RANDOMIZED_WITHIN_STUDY,
    )
    with pytest.raises(ValidationError, match="timed exposure"):
        revalidate(
            manifest,
            capabilities=no_intervention,
            claim_eligibility=(selection,),
        )
    with pytest.raises(ValidationError, match="must remain conditional"):
        eligible(
            ScientificClaim.INTERVENTION_EFFECT,
            scope(interventions=(term("drug"),), horizons=(1.0,)),
            basis=IdentificationBasis.QUASI_EXPERIMENTAL,
        )


def test_timed_causal_claims_require_recorded_event_ordering() -> None:
    manifest = manifest_factory()
    timing = manifest.capabilities.timing.model_copy(update={"event_ordering_recorded": False})
    capabilities = manifest.capabilities.model_copy(update={"timing": timing})
    with pytest.raises(ValidationError, match="recorded event ordering"):
        revalidate(manifest, capabilities=capabilities)


def test_intervention_functional_endpoint_must_cover_claim_horizon() -> None:
    manifest = manifest_factory()
    too_early = manifest.capabilities.functional.outputs[0].model_copy(
        update={"measurement_time_seconds": 3600.0}
    )
    capabilities = manifest.capabilities.model_copy(
        update={"functional": FunctionalCapability(outputs=(too_early,))}
    )
    intervention_claim = manifest.claim_eligibility[1]
    with pytest.raises(ValidationError, match="does not cover the claim horizon"):
        revalidate(
            manifest,
            capabilities=capabilities,
            claim_eligibility=(intervention_claim,),
        )


def test_counterfactual_generalization_fails_without_structured_transport_scope() -> None:
    manifest = manifest_factory()
    counterfactual = eligible(
        ScientificClaim.COUNTERFACTUAL_GENERALIZATION,
        manifest.claim_eligibility[1].scope,
        basis=IdentificationBasis.TRANSPORTED_UNDER_ASSUMPTIONS,
        status=ClaimStatus.CONDITIONALLY_ELIGIBLE,
    )
    with pytest.raises(ValidationError, match="structured transport scope"):
        revalidate(manifest, claim_eligibility=(counterfactual,))


def test_measurement_evidence_cannot_be_documentation_only_or_mislabeled() -> None:
    manifest = manifest_factory()
    documentation_modality = manifest.capabilities.modalities[0].model_copy(
        update={
            "source_ids": ("documentation",),
            "raw_available": False,
            "processed_available": True,
        }
    )
    capabilities = manifest.capabilities.model_copy(
        update={"modalities": (documentation_modality, manifest.capabilities.modalities[1])}
    )
    with pytest.raises(ValidationError, match="raw or processed artifacts"):
        revalidate(manifest, capabilities=capabilities)
    wrong_availability = manifest.capabilities.modalities[0].model_copy(
        update={"raw_available": False}
    )
    capabilities = manifest.capabilities.model_copy(
        update={"modalities": (wrong_availability, manifest.capabilities.modalities[1])}
    )
    with pytest.raises(ValidationError, match="availability must match"):
        revalidate(manifest, capabilities=capabilities)


def test_spatial_claim_requires_cell_resolved_evidence_and_boundary() -> None:
    manifest = manifest_factory()
    capabilities = manifest.capabilities.model_copy(
        update={
            "spatial": SpatialCapability(
                resolution=SpatialResolution.SAMPLE_REGION,
                source_ids=("metadata",),
                regions_recorded=True,
            )
        }
    )
    spatial_claim = eligible(
        ScientificClaim.SPATIAL_CONTEXT,
        scope(system_boundary=SystemBoundary.SPATIAL_TISSUE_NICHE),
        basis=IdentificationBasis.ASSOCIATIONAL,
    )
    with pytest.raises(ValidationError, match="cell-resolved evidence"):
        revalidate(manifest, capabilities=capabilities, claim_eligibility=(spatial_claim,))


def test_spatial_claim_requires_a_modality_linked_to_spatial_alignment() -> None:
    manifest = manifest_factory()
    capabilities = manifest.capabilities.model_copy(
        update={
            "spatial": SpatialCapability(
                resolution=SpatialResolution.CELL_COORDINATES,
                source_ids=("metadata",),
                coordinate_dimensions=2,
                subject_alignment=SubjectAlignment.SAME_CELL,
                alignment_group="spatial-cells",
                alignment_key_field="cell_id",
            )
        }
    )
    spatial_claim = eligible(
        ScientificClaim.SPATIAL_CONTEXT,
        scope(
            system_boundary=SystemBoundary.SPATIAL_TISSUE_NICHE,
            modalities=(term("transcriptome"),),
        ),
        basis=IdentificationBasis.ASSOCIATIONAL,
    )
    with pytest.raises(ValidationError, match="aligned to the spatial evidence"):
        revalidate(manifest, capabilities=capabilities, claim_eligibility=(spatial_claim,))

    spatial_modality = manifest.capabilities.modalities[0].model_copy(
        update={
            "subject_alignment": SubjectAlignment.SAME_CELL,
            "alignment_group": "spatial-cells",
            "alignment_key_field": "cell_id",
        }
    )
    linked_capabilities = capabilities.model_copy(
        update={
            "modalities": (spatial_modality, manifest.capabilities.modalities[1]),
        }
    )
    assert revalidate(
        manifest,
        capabilities=linked_capabilities,
        claim_eligibility=(spatial_claim,),
    ).claim_eligibility == (spatial_claim,)


def test_environment_and_sampling_timing_assertions_are_cross_checked() -> None:
    manifest = manifest_factory()
    no_environment_time = manifest.capabilities.timing.model_copy(
        update={"environment_times_recorded": False}
    )
    capabilities = manifest.capabilities.model_copy(update={"timing": no_environment_time})
    with pytest.raises(ValidationError, match="time-resolved environment"):
        revalidate(manifest, capabilities=capabilities)
    no_observation_time = manifest.capabilities.timing.model_copy(
        update={"observation_times_recorded": False, "timepoints_seconds": ()}
    )
    capabilities = manifest.capabilities.model_copy(update={"timing": no_observation_time})
    with pytest.raises(ValidationError, match="sampling time field"):
        revalidate(manifest, capabilities=capabilities)


def test_invalid_model_instances_are_revalidated_at_every_manifest_boundary() -> None:
    manifest = manifest_factory()
    invalid_manifest = manifest.model_copy(update={"schema_version": "1.0"})
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(invalid_manifest)

    invalid_modality = manifest.capabilities.modalities[0].model_copy(
        update={"alignment_key_field": "cell_id"}
    )
    invalid_capabilities = manifest.capabilities.model_copy(
        update={"modalities": (invalid_modality, manifest.capabilities.modalities[1])}
    )
    invalid_manifest = manifest.model_copy(update={"capabilities": invalid_capabilities})
    with pytest.raises(ValidationError, match="alignment key must match"):
        DatasetManifest.model_validate(invalid_manifest)


def test_split_safety_uses_sampling_subject_even_without_replicate_metadata() -> None:
    manifest = manifest_factory()
    unsafe = manifest.experimental_design.model_copy(
        update={
            "default_split_unit": ExperimentalUnitLevel.CELL,
            "biological_replicate_unit": None,
            "randomization_unit": None,
        }
    )
    with pytest.raises(ValidationError, match="default split unit cannot be finer"):
        revalidate(manifest, experimental_design=unsafe)


def test_supported_dynamics_horizon_must_fit_observed_timepoints() -> None:
    manifest = manifest_factory()
    overlong = eligible(
        ScientificClaim.POPULATION_DYNAMICS,
        scope(modalities=(term("transcriptome"),), horizons=(315_360_000.0,)),
        basis=IdentificationBasis.ASSOCIATIONAL,
    )
    with pytest.raises(ValidationError, match="horizon exceeds observed temporal support"):
        revalidate(manifest, claim_eligibility=(overlong,))


def test_supported_dynamics_cutoff_must_be_observed() -> None:
    manifest = manifest_factory()
    claim_scope = scope(modalities=(term("transcriptome"),), horizons=(3600.0,)).model_copy(
        update={"inference_cutoff_seconds": -3600.0}
    )
    claim = eligible(
        ScientificClaim.POPULATION_DYNAMICS,
        claim_scope,
        basis=IdentificationBasis.ASSOCIATIONAL,
    )
    with pytest.raises(ValidationError, match="inference cutoff lies outside"):
        revalidate(manifest, claim_eligibility=(claim,))


def test_supported_claim_evidence_must_reach_each_scoped_capability() -> None:
    manifest = manifest_factory()
    claim_scope = scope(modalities=(term("transcriptome"),), horizons=(86400.0,))
    disconnected = ClaimEligibility(
        claim=ScientificClaim.POPULATION_DYNAMICS,
        status=ClaimStatus.ELIGIBLE,
        identification_basis=IdentificationBasis.ASSOCIATIONAL,
        scope=claim_scope,
        evidence_source_ids=("raw",),
        evidence_notes=("Raw counts alone do not establish the timing structure.",),
    )
    with pytest.raises(ValidationError, match="scoped timing"):
        revalidate(manifest, claim_eligibility=(disconnected,))


def test_ungated_and_unconditioned_supported_claims_fail_closed() -> None:
    manifest = manifest_factory()
    for claim in (
        ScientificClaim.ASSAY_MEASUREMENT_MODEL,
        ScientificClaim.SNAPSHOT_STATE_PRIOR,
    ):
        with pytest.raises(ValidationError, match=r"scoped|measurement modality"):
            revalidate(manifest, claim_eligibility=(eligible(claim, scope()),))

    counterfactual = eligible(
        ScientificClaim.COUNTERFACTUAL_GENERALIZATION,
        scope(),
        basis=IdentificationBasis.TRANSPORTED_UNDER_ASSUMPTIONS,
        status=ClaimStatus.CONDITIONALLY_ELIGIBLE,
    )
    with pytest.raises(ValidationError, match="structured transport scope"):
        revalidate(manifest, claim_eligibility=(counterfactual,))

    functional = eligible(
        ScientificClaim.FUNCTIONAL_OUTCOME,
        scope(outputs=(term("cell viability"),), horizons=(86400.0,)),
        basis=IdentificationBasis.RANDOMIZED_WITHIN_STUDY,
    )
    with pytest.raises(ValidationError, match="conditioning modalities"):
        revalidate(manifest, claim_eligibility=(functional,))


def test_functional_units_and_field_clock_claims_fail_closed() -> None:
    manifest = manifest_factory()
    undeclared = manifest.capabilities.functional.outputs[0].model_copy(
        update={
            "aggregation_level": ExperimentalUnitLevel.DONOR,
            "alignment_key_field": "donor_id",
        }
    )
    capabilities = manifest.capabilities.model_copy(
        update={"functional": FunctionalCapability(outputs=(undeclared,))}
    )
    with pytest.raises(ValidationError, match="undeclared unit level"):
        revalidate(manifest, capabilities=capabilities)

    disconnected_unit = ExperimentalUnitSpec(
        level=ExperimentalUnitLevel.DONOR,
        id_field="donor_id",
        source_ids=("metadata",),
    )
    design = manifest.experimental_design.model_copy(
        update={"units": (*manifest.experimental_design.units, disconnected_unit)}
    )
    disconnected = manifest.capabilities.functional.outputs[0].model_copy(
        update={
            "aggregation_level": ExperimentalUnitLevel.DONOR,
            "alignment_key_field": "donor_id",
        }
    )
    capabilities = manifest.capabilities.model_copy(
        update={"functional": FunctionalCapability(outputs=(disconnected,))}
    )
    with pytest.raises(ValidationError, match="sampling unit or one of its declared ancestors"):
        revalidate(manifest, experimental_design=design, capabilities=capabilities)

    field_timed = manifest.capabilities.functional.outputs[0].model_copy(
        update={"measurement_time_seconds": None, "measurement_time_field": "outcome_time"}
    )
    capabilities = manifest.capabilities.model_copy(
        update={"functional": FunctionalCapability(outputs=(field_timed,))}
    )
    claim_scope = ClaimScope(
        subject_kind=SamplingSubjectKind.POPULATION,
        system_boundary=SystemBoundary.POPULATION,
        biological_systems=(term("cultured cell population"),),
        modalities=(term("transcriptome"),),
        functional_outputs=(term("cell viability"),),
        horizons_seconds=(86400.0,),
        inference_cutoff_field="prediction_time",
    )
    claim = eligible(
        ScientificClaim.FUNCTIONAL_OUTCOME,
        claim_scope,
        basis=IdentificationBasis.ASSOCIATIONAL,
        status=ClaimStatus.CONDITIONALLY_ELIGIBLE,
    )
    with pytest.raises(ValidationError, match="field-clock future outcomes are unsupported"):
        revalidate(manifest, capabilities=capabilities, claim_eligibility=(claim,))


def test_individual_dynamics_rejects_any_destructive_conditioning_modality() -> None:
    manifest = manifest_factory()
    sampling = SamplingDesign(
        subject_kind=SamplingSubjectKind.INDIVIDUAL_CELL,
        subject_unit=ExperimentalUnitLevel.CELL,
        subject_id_field="cell_id",
        source_ids=("processed",),
        mode=SamplingMode.LONGITUDINAL_NONDESTRUCTIVE,
        linkage=SubjectLinkage.SAME_CELL,
        time_field="time",
        source_time_units="s",
    )
    design = manifest.experimental_design.model_copy(update={"sampling": sampling})
    live = manifest.capabilities.modalities[0].model_copy(
        update={
            "subject_alignment": SubjectAlignment.SAME_CELL,
            "alignment_group": "cell-live",
            "alignment_key_field": "cell_id",
            "destructive": False,
        }
    )
    destructive = manifest.capabilities.modalities[1].model_copy(
        update={
            "subject_alignment": SubjectAlignment.SAME_CELL,
            "alignment_group": "cell-live",
            "alignment_key_field": "cell_id",
            "destructive": True,
        }
    )
    capabilities = manifest.capabilities.model_copy(update={"modalities": (live, destructive)})
    claim = eligible(
        ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS,
        scope(
            subject_kind=SamplingSubjectKind.INDIVIDUAL_CELL,
            system_boundary=SystemBoundary.ISOLATED_CELL,
            modalities=(term("transcriptome"), term("surface proteome")),
            horizons=(86400.0,),
        ),
        basis=IdentificationBasis.ASSOCIATIONAL,
    )
    with pytest.raises(ValidationError, match="nondestructive same-cell evidence"):
        revalidate(
            manifest,
            experimental_design=design,
            capabilities=capabilities,
            claim_eligibility=(claim,),
        )


def test_assay_measurement_claim_must_cite_raw_modality_evidence() -> None:
    manifest = manifest_factory()
    claim = eligible(
        ScientificClaim.ASSAY_MEASUREMENT_MODEL,
        scope(modalities=(term("transcriptome"),)),
    )
    with pytest.raises(ValidationError, match="must cite each scoped modality's raw source"):
        revalidate(manifest, claim_eligibility=(claim,))

    raw_backed = claim.model_copy(update={"evidence_source_ids": ("raw",)})
    assert revalidate(manifest, claim_eligibility=(raw_backed,)).claim_eligibility == (raw_backed,)
