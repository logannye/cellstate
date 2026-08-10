#!/usr/bin/env python3
"""Build the frozen sci-Plex3 K562 Vertical A component benchmark.

This generator consumes only checked-in, content-addressed preparation artifacts.  It does not
open the 2.5 GB H5AD, fit a model, score a baseline, or inspect held-out expression values.  The
large membership arrays were produced by ``prepare_sciplex3_k562.py`` from exact source bytes;
this script binds those arrays to the reviewed manifest, schema-v2 query, split, metric, leakage,
and component-admission contracts.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from cellstate.data.benchmarks import (
    AcceptanceGroupOperator,
    AuditCheckStatus,
    BaselineApplicabilityRule,
    BaselineDefinitionReference,
    BaselineMarginMode,
    BaselineRequirement,
    BaselineRun,
    BaselineRunStatus,
    BenchmarkAcceptanceGroup,
    BenchmarkAcceptancePolicy,
    BenchmarkAcceptanceRule,
    BenchmarkAdmission,
    BenchmarkAdmissionStatus,
    BenchmarkArtifact,
    BenchmarkBaselineDefinition,
    BenchmarkDefinition,
    BenchmarkEvaluationCase,
    BenchmarkEvaluationCaseSet,
    BenchmarkEvidenceBinding,
    BenchmarkIntent,
    BenchmarkLeakageAudit,
    BenchmarkLifecycle,
    BenchmarkMetricDefinition,
    BenchmarkParameter,
    BenchmarkPartition,
    BenchmarkPartitionRole,
    BenchmarkScope,
    BenchmarkSplitPlan,
    BestApplicableBaselineComparator,
    CanonicalIdMembership,
    ContentAddressedArtifact,
    EvaluationCasePartitionBinding,
    EvaluationCaseRole,
    EvaluationContextBinding,
    EvaluationInterventionMultiplicity,
    EvidenceInterventionMapping,
    EvidenceResolutionBinding,
    EvidenceScopeBinding,
    EvidenceTargetMapping,
    ExactBaselineComparator,
    ExperimentalUnitBinding,
    ExplicitPartitionMembership,
    LeakageAuditCheck,
    LeakageCheckKind,
    LossAssessmentIdentity,
    MetricAggregation,
    MetricAssessmentIdentity,
    MetricDefinitionReference,
    MetricDependenceKind,
    MetricDependenceUnit,
    MetricDirection,
    MetricMissingnessPolicy,
    MetricResamplingScheme,
    MetricUncertaintySpec,
    MetricWeightingPolicy,
    MetricWeightingScheme,
    PartitionUniverse,
    PredictionRepresentation,
    ProtectedGroupBinding,
    ProtectedGroupClosure,
    ProtectedGroupMembership,
    ProtectedGroupReason,
    SpecificationOnlyImplementationBinding,
    StateQueryBinding,
    TargetRepresentation,
    ThresholdComparison,
    ThresholdEstimate,
    VersionedImplementation,
    verify_benchmark_artifact,
)
from cellstate.data.manifests import (
    AccessMode,
    AssessmentKind,
    AssessmentScope,
    ClaimAssessment,
    ClaimAssessmentReference,
    CohortSelectionStage,
    CompositeIdentityEncoding,
    ControlPredicateValueType,
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
    IdentificationBasis,
    InterventionCapability,
    LineageCapability,
    LossEligibilityAssessment,
    MatchedControlDefinition,
    MatchedControlPredicate,
    MetricEligibilityAssessment,
    MetricFamily,
    MetricPartitionPurpose,
    ModalitySpec,
    PermissionStatus,
    PublicRealDataOrigin,
    RandomizedEndpointContrast,
    RealizationEvidence,
    SamplingDesign,
    SamplingMode,
    SamplingSubjectKind,
    ScientificClaim,
    SourceArtifact,
    SourceKind,
    SpatialCapability,
    SubjectAlignment,
    SubjectLinkage,
    TimingCapability,
    UnitIdentityExpression,
    UnitIdentityExpressionKind,
    UsePermission,
)
from cellstate.data.manifests import (
    AssignmentMechanism as ManifestAssignmentMechanism,
)
from cellstate.domain.common import CausalStatus, OntologyTerm, canonical_fingerprint
from cellstate.domain.events import (
    AssignmentMechanism,
    CollectionEffect,
    EvidenceRole,
    MissingnessStatus,
    ObservationCollection,
    PerturbationStatus,
    ReversibilityStatus,
    ScheduleKind,
)
from cellstate.domain.query import (
    AcceptanceThresholds,
    AssayPurpose,
    AssaySpec,
    EvidencePolicy,
    FutureAssayObservationEndpoint,
    IntegerRange,
    InterventionSpec,
    NumericDomain,
    OutputSpec,
    PredictionHorizon,
    QueryConstraints,
    RealizationEvidenceRequirement,
    ScalarRange,
    ScheduleDomain,
    StateQuery,
    SystemBoundary,
    TargetCensoringPolicy,
    TargetCensoringSemantics,
    TargetMissingnessPolicy,
    TargetMissingnessSemantics,
    Timescale,
    VersionedReference,
)
from cellstate.domain.subjects import (
    AggregationStatistic,
    IdentityBasis,
    SubjectKind,
    SubjectSpecification,
    TargetAggregation,
)
from cellstate.training.objectives import LossKind

REPO_ROOT = Path(__file__).resolve().parents[1]
PREP_DIR = REPO_ROOT / "benchmarks/artifacts/sciplex3-k562-24h-v1"
OUTPUT_DIR = REPO_ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1"
SUPPORT_DIR = OUTPUT_DIR / "support"
MANIFEST_PATH = REPO_ROOT / "data_manifests/reviewed/sciplex3-k562-24h.json"
QUERY_PATH = OUTPUT_DIR / "state-query.json"
BENCHMARK_PATH = OUTPUT_DIR / "benchmark-artifact.json"
RAW_BASE = "https://raw.githubusercontent.com/logannye/cellstate/main"
BENCHMARK_ID = "vertical-a.sciplex3-k562-24h-replicate-transfer.v1"
QUERY_ID = "vertical-a.k562.small-molecule.context-to-24h.v1"
SOURCE_ID = "scperturb-v1.4-sciplex3-h5ad"
PHYSICAL_DATASET_BINDING_ID = "sciplex3-k562-24h-physical-source"
PAPER_SOURCE_ID = "srivatsan-2019-primary-paper"
PDATA_SOURCE_ID = "geo-gsm4150378-sciplex3-pdata"
PROCESSING_SOURCE_ID = "scperturb-pinned-sciplex3-processing"
HORIZON_NAME = "24h-endpoint"
TARGET_TERM = OntologyTerm(
    label=(
        "raw sci-Plex3 K562 nuclear RNA UMI count-vector population distribution on the "
        "ordered train-derived 2,000-feature panel"
    ),
    identifier="CELLSTATE:sciplex3-k562-24h-train-2000-raw-umi-distribution",
    namespace="CELLSTATE",
)
RNA_MODALITY = OntologyTerm(
    label="single nucleus RNA sequencing",
    identifier="EFO:0009809",
    namespace="EFO",
)
CONTEXT_MODALITY = OntologyTerm(
    label="declared K562 well design context",
    identifier="CELLSTATE:sciplex3-k562-design-context",
    namespace="CELLSTATE",
)
K562 = OntologyTerm(label="K562", identifier="CVCL:0004", namespace="CVCL")
HUMAN = OntologyTerm(label="Homo sapiens", identifier="NCBITaxon:9606", namespace="NCBITaxon")
DOSES_NM = (10, 100, 1000, 10000)
MANDATORY_PROBABILISTIC_BASELINE_IDS = (
    "exact-condition-negative-binomial",
    "exact-condition-rep1-empirical-resampling",
    "hierarchical-well-negative-binomial",
    "low-rank-compound-dose-response",
    "matched-vehicle-resampling",
)
ROLES = (
    "train",
    "calibration",
    "model_selection_validation",
    "untouched_test",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_canonical(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = getattr(value, "canonical_json_bytes", None)
    path.write_bytes(payload if isinstance(payload, bytes) else _canonical_bytes(value))


def _write_pretty(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _repo_uri(path: Path) -> str:
    return f"{RAW_BASE}/{path.relative_to(REPO_ROOT).as_posix()}"


def _artifact_for_file(
    artifact_id: str,
    path: Path,
    *,
    media_type: str = "application/json",
) -> ContentAddressedArtifact:
    payload = path.read_bytes()
    return ContentAddressedArtifact(
        artifact_id=artifact_id,
        uri=_repo_uri(path),
        sha256=_sha256_bytes(payload),
        byte_count=len(payload),
        media_type=media_type,
    )


def _membership_artifact(role: str, name: str) -> ContentAddressedArtifact:
    path = PREP_DIR / "memberships" / f"{role}-{name}.json"
    return _artifact_for_file(f"sciplex3-{role}-{name}", path)


def _identity_field(name: str) -> UnitIdentityExpression:
    return UnitIdentityExpression(
        kind=UnitIdentityExpressionKind.SOURCE_FIELD,
        source_fields=(name,),
    )


def _identity_composite(*names: str) -> UnitIdentityExpression:
    return UnitIdentityExpression(
        kind=UnitIdentityExpressionKind.COMPOSITE_SOURCE_FIELDS,
        source_fields=names,
        composite_encoding=CompositeIdentityEncoding.CANONICAL_JSON_UTF8_STRING_ARRAY_V1,
    )


CULTURE_IDENTITY = UnitIdentityExpression(
    kind=UnitIdentityExpressionKind.MANIFEST_CONSTANT,
    constant_value="sciplex3-k562-study-stratum",
)
SAMPLE_IDENTITY = _identity_field("replicate")
PLATE_IDENTITY = _identity_field("plate")
WELL_IDENTITY = _identity_composite("plate", "well")
CELL_IDENTITY = _identity_field("_index")
CONDITION_IDENTITY = _identity_field("source_scoped_condition_id")
COMPOUND_IDENTITY = _identity_field("normalized_perturbation_label")


def _unit(level: ExperimentalUnitLevel) -> ExperimentalUnitBinding:
    identities = {
        ExperimentalUnitLevel.CULTURE: CULTURE_IDENTITY,
        ExperimentalUnitLevel.SAMPLE: SAMPLE_IDENTITY,
        ExperimentalUnitLevel.PLATE: PLATE_IDENTITY,
        ExperimentalUnitLevel.WELL: WELL_IDENTITY,
        ExperimentalUnitLevel.CELL: CELL_IDENTITY,
    }
    return ExperimentalUnitBinding(level=level, identity=identities[level])


def _all_permissions(
    status: PermissionStatus, reason: str | None = None
) -> tuple[UsePermission, ...]:
    conditions = (
        () if status is PermissionStatus.PERMITTED else (reason or "Legal review required.",)
    )
    return tuple(
        UsePermission(use_case=use_case, status=status, conditions=conditions)
        for use_case in DataUseCase
    )


def _compound_kind(label: str) -> OntologyTerm:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return OntologyTerm(
        label=f"sci-Plex3 source-scoped administered agent: {label}",
        identifier=f"SCIPLEX3ACTION:{digest}",
        namespace="SCIPLEX3ACTION",
    )


def _assessment_scope(labels: tuple[str, ...]) -> AssessmentScope:
    return AssessmentScope(
        subject_kind=SamplingSubjectKind.POPULATION,
        system_boundary=SystemBoundary.POPULATION,
        biological_systems=(K562,),
        modalities=(RNA_MODALITY,),
        intervention_kinds=tuple(_compound_kind(label) for label in labels),
        horizons_seconds=(86400.0,),
        inference_cutoff_seconds=0.0,
    )


def _negative_claim(claim: ScientificClaim, blocker: str) -> ClaimAssessment:
    return ClaimAssessment(
        assessment_id=f"sciplex3-k562-{claim.value.replace('_', '-')}",
        claim=claim,
        status=(
            EligibilityStatus.NOT_ASSESSED
            if claim is ScientificClaim.ASSAY_MEASUREMENT_MODEL
            else EligibilityStatus.INELIGIBLE
        ),
        identification_basis=IdentificationBasis.NONE,
        scope=AssessmentScope(
            subject_kind=SamplingSubjectKind.POPULATION,
            system_boundary=SystemBoundary.POPULATION,
            biological_systems=(K562,),
            modalities=(RNA_MODALITY,),
        ),
        blockers=(blocker,),
    )


def build_manifest(prep: dict[str, Any]) -> DatasetManifest:
    labels = prep["labels"]
    compound_kinds = tuple(_compound_kind(label) for label in labels)
    scope = _assessment_scope(labels)
    intervention_claim = ClaimAssessment(
        assessment_id="sciplex3-k562-randomized-endpoint-intervention-effect",
        claim=ScientificClaim.INTERVENTION_EFFECT,
        status=EligibilityStatus.ELIGIBLE,
        identification_basis=IdentificationBasis.RANDOMIZED_WITHIN_STUDY,
        scope=scope,
        evidence_source_ids=(PAPER_SOURCE_ID, PDATA_SOURCE_ID, PROCESSING_SOURCE_ID, SOURCE_ID),
        execution_source_ids=(SOURCE_ID,),
        evidence_notes=(
            "The primary paper states that 188 compounds at four exact doses were randomized "
            "across well positions in duplicate culture plates and applied for 24 hours.",
            "The corrected scPerturb 1.4 H5AD preserves plate, well, replicate, compound, dose, "
            "control, time, K562 identity, and integer UMI endpoint records.",
            "The identified reference contrast is assignment intention versus same-plate vehicle "
            "among recovered nuclei; it does not identify target engagement, survival, or "
            "transport.",
        ),
    )
    snapshot_scope = AssessmentScope(
        subject_kind=SamplingSubjectKind.POPULATION,
        system_boundary=SystemBoundary.POPULATION,
        biological_systems=(K562,),
        modalities=(RNA_MODALITY,),
        intervention_kinds=compound_kinds,
    )
    snapshot_claim = ClaimAssessment(
        assessment_id="sciplex3-k562-endpoint-snapshot-prior",
        claim=ScientificClaim.SNAPSHOT_STATE_PRIOR,
        status=EligibilityStatus.ELIGIBLE,
        identification_basis=IdentificationBasis.DESCRIPTIVE,
        scope=snapshot_scope,
        evidence_source_ids=(PAPER_SOURCE_ID, PDATA_SOURCE_ID, PROCESSING_SOURCE_ID, SOURCE_ID),
        execution_source_ids=(SOURCE_ID,),
        evidence_notes=(
            "The exact K562 slice provides destructive 24-hour endpoint UMI-count distributions "
            "by independently treated well.",
            "This descriptive role is confined to recovered-nucleus assay distributions and is "
            "not a pretreatment or latent living-cell state claim.",
        ),
    )
    claim_ref = ClaimAssessmentReference(
        assessment_id=intervention_claim.assessment_id,
        assessment_fingerprint=intervention_claim.fingerprint,
    )
    metric_specs = (
        (
            "sciplex3.marginal-crps-logcp10k",
            MetricFamily.PREDICTIVE_PROPER_SCORE,
        ),
        (
            "sciplex3.joint-energy-train-pca",
            MetricFamily.PREDICTIVE_PROPER_SCORE,
        ),
        (
            "sciplex3.vehicle-relative-pseudobulk-rmse",
            MetricFamily.INTERVENTION_EFFECT,
        ),
        (
            "sciplex3.marginal-coverage-error-p50",
            MetricFamily.CALIBRATION,
        ),
        (
            "sciplex3.marginal-coverage-error-p80",
            MetricFamily.CALIBRATION,
        ),
        (
            "sciplex3.marginal-coverage-error-p95",
            MetricFamily.CALIBRATION,
        ),
        (
            "sciplex3.marginal-interval-width-p50",
            MetricFamily.CALIBRATION,
        ),
        (
            "sciplex3.marginal-interval-width-p80",
            MetricFamily.CALIBRATION,
        ),
        (
            "sciplex3.marginal-interval-width-p95",
            MetricFamily.CALIBRATION,
        ),
        (
            "sciplex3.four-dose-profile-diagnostic",
            MetricFamily.INTERVENTION_EFFECT,
        ),
    )
    loss = LossEligibilityAssessment(
        assessment_id="sciplex3-k562-intervention-effect-loss",
        status=EligibilityStatus.ELIGIBLE,
        scope=scope,
        required_split_unit=ExperimentalUnitLevel.PLATE,
        data_source_ids=(SOURCE_ID,),
        supporting_claim_assessments=(claim_ref,),
        evidence_notes=(
            "Whole protected plates can be separated while wells remain the independent "
            "assignment and endpoint population units.",
            "The exact source has a randomized assignment-to-24-hour endpoint contrast and "
            "matched same-plate controls without a fabricated baseline assay.",
        ),
        loss_kind=LossKind.INTERVENTION_EFFECT,
    )
    metrics = tuple(
        MetricEligibilityAssessment(
            assessment_id=f"{metric_id}-untouched-test",
            status=EligibilityStatus.ELIGIBLE,
            scope=scope,
            required_split_unit=ExperimentalUnitLevel.PLATE,
            data_source_ids=(SOURCE_ID,),
            supporting_claim_assessments=(claim_ref,),
            evidence_notes=(
                "The frozen whole-plate partition preserves well-level replication and an "
                "untouched replicate-transfer test.",
                "Metric inference must weight wells or predeclared condition groups, never "
                "recovered nuclei as independent replicates.",
            ),
            metric_id=metric_id,
            metric_family=family,
            partition_purpose=MetricPartitionPurpose.UNTOUCHED_TEST,
        )
        for metric_id, family in metric_specs
    )
    negatives = (
        _negative_claim(
            ScientificClaim.ASSAY_MEASUREMENT_MODEL,
            "The executable artifact is a corrected processed H5AD and no raw-read-to-UMI "
            "measurement-error model is bound.",
        ),
        _negative_claim(
            ScientificClaim.SAME_CELL_MULTIMODAL_FUSION,
            "The endpoint exposes one destructive RNA modality and no same-cell second modality.",
        ),
        _negative_claim(
            ScientificClaim.SAMPLE_LEVEL_MULTIMODAL_FUSION,
            "The endpoint exposes one molecular modality and no independent sample-aligned "
            "modality pair.",
        ),
        _negative_claim(
            ScientificClaim.POPULATION_DYNAMICS,
            "There is one destructive 24-hour population endpoint and no linked repeated "
            "observation of a well population.",
        ),
        _negative_claim(
            ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS,
            "Recovered nuclei are destructively sampled once and are neither tracked nor "
            "measured before treatment.",
        ),
        _negative_claim(
            ScientificClaim.COUNTERFACTUAL_GENERALIZATION,
            "No structured external target domain or transport evidence is bound.",
        ),
        _negative_claim(
            ScientificClaim.LINEAGE_FATE,
            "No lineage identity, parentage, or future fate readout is present.",
        ),
        _negative_claim(
            ScientificClaim.SPATIAL_CONTEXT,
            "No cell-resolved spatial coordinates, neighborhood graph, or tissue niche is present.",
        ),
        _negative_claim(
            ScientificClaim.FUNCTIONAL_OUTCOME,
            "No well-linked future functional output is present in the executable benchmark "
            "artifact.",
        ),
        _negative_claim(
            ScientificClaim.RETROSPECTIVE_INTERVENTION_SELECTION,
            "No decision utility, safety constraint outcome, or externally validated selection "
            "policy is bound.",
        ),
    )
    universe = prep["universe"]
    sources = (
        SourceArtifact(
            source_id=SOURCE_ID,
            kind=SourceKind.PROCESSED,
            uri="https://zenodo.org/api/records/13350497/files/SrivatsanTrapnell2020_sciplex3.h5ad/content",
            sha256="603ed16c5e25401c8a7f5bb0b2b045179701017d65dcfc6aeea71722a66cd10a",
            media_type="application/x-hdf5",
            access_mode=AccessMode.OPEN_DOWNLOAD,
            accession="10.5281/zenodo.13350497",
            release="1.4",
            parent_study_accession="GSE139944",
            parent_study_release="2019-12-05",
            byte_count=2526631614,
            retrieved_at=datetime(2026, 8, 9, 19, 21, 39, tzinfo=UTC),
        ),
        SourceArtifact(
            source_id=PDATA_SOURCE_ID,
            kind=SourceKind.METADATA,
            uri="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150378/suppl/GSM4150378_sciPlex3_pData.txt.gz",
            sha256="860ba5c21846e2cfe97c39c805895982ef9db4b108cdf8b841a4a91be65024c2",
            media_type="application/gzip",
            access_mode=AccessMode.OPEN_DOWNLOAD,
            accession="GSM4150378",
            release="2019-12-05",
            parent_study_accession="GSE139944",
            parent_study_release="2019-12-05",
            byte_count=84673357,
            retrieved_at=datetime(2026, 8, 9, 19, 2, 42, tzinfo=UTC),
        ),
        SourceArtifact(
            source_id=PAPER_SOURCE_ID,
            kind=SourceKind.DOCUMENTATION,
            uri="https://cole-trapnell-lab.github.io/pdfs/papers/srivatsan_mcfaline_ramani_sci-plex_Science_2019.pdf",
            sha256="cb9efb6c90af5c4ecc744e7235ec8bbdd4cb7971e75c16afb7946b8bbf16af93",
            media_type="application/pdf",
            access_mode=AccessMode.OPEN_DOWNLOAD,
            accession="doi:10.1126/science.aax6234",
            release="2019-12-05",
            parent_study_accession="GSE139944",
            parent_study_release="2019-12-05",
            byte_count=1534850,
            retrieved_at=datetime(2026, 8, 9, tzinfo=UTC),
        ),
        SourceArtifact(
            source_id=PROCESSING_SOURCE_ID,
            kind=SourceKind.DOCUMENTATION,
            uri="https://raw.githubusercontent.com/sanderlab/scPerturb/fd4f5b360233520b0fc297f1bdf8953a6bee2dbb/dataset_processing/scripts/SrivatsanTrapnell2020.py",
            sha256="1d216f8f09a758682f7635a7509a3c359842a17925995424b1041a80cd2fe386",
            media_type="text/x-python",
            access_mode=AccessMode.OPEN_DOWNLOAD,
            accession="github:sanderlab/scPerturb@fd4f5b360233520b0fc297f1bdf8953a6bee2dbb",
            release="fd4f5b360233520b0fc297f1bdf8953a6bee2dbb",
            parent_study_accession="GSE139944",
            parent_study_release="2019-12-05",
            byte_count=7337,
            retrieved_at=datetime(2026, 8, 9, tzinfo=UTC),
        ),
    )
    unknown_geo = (
        "GEO public access does not itself resolve downstream copyright permission; this "
        "artifact is review evidence only and is not executable benchmark input."
    )
    unknown_paper = (
        "The publicly readable author manuscript is review evidence only; downstream copying "
        "permissions require separate legal review."
    )
    policies = (
        DataUsePolicy(
            policy_id="scperturb-v1.4-cc-by-4.0",
            source_ids=(SOURCE_ID,),
            license_name="Creative Commons Attribution 4.0 International",
            terms_url="https://creativecommons.org/licenses/by/4.0/",
            reviewed_on=date(2026, 8, 9),
            spdx_identifier="CC-BY-4.0",
            permissions=_all_permissions(PermissionStatus.PERMITTED),
            attribution_requirements=(
                "Cite Srivatsan et al., DOI 10.1126/science.aax6234, and scPerturb Zenodo release "
                "1.4; retain a CC BY 4.0 link and indicate changes.",
            ),
        ),
        DataUsePolicy(
            policy_id="geo-review-only-rights-unresolved",
            source_ids=(PDATA_SOURCE_ID,),
            license_name="GEO public deposit with submitter rights unresolved",
            terms_url="https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html",
            reviewed_on=date(2026, 8, 9),
            permissions=_all_permissions(PermissionStatus.UNKNOWN, unknown_geo),
            attribution_requirements=("Cite GEO GSE139944 and GSM4150378.",),
        ),
        DataUsePolicy(
            policy_id="primary-paper-review-only-rights-unresolved",
            source_ids=(PAPER_SOURCE_ID,),
            license_name="Public author manuscript with downstream rights unresolved",
            terms_url="https://www.science.org/doi/10.1126/science.aax6234",
            reviewed_on=date(2026, 8, 9),
            permissions=_all_permissions(PermissionStatus.UNKNOWN, unknown_paper),
            attribution_requirements=("Cite Srivatsan et al., DOI 10.1126/science.aax6234.",),
        ),
        DataUsePolicy(
            policy_id="scperturb-processing-mit",
            source_ids=(PROCESSING_SOURCE_ID,),
            license_name="MIT License",
            terms_url="https://raw.githubusercontent.com/sanderlab/scPerturb/fd4f5b360233520b0fc297f1bdf8953a6bee2dbb/LICENSE",
            reviewed_on=date(2026, 8, 9),
            spdx_identifier="MIT",
            permissions=_all_permissions(PermissionStatus.PERMITTED),
            attribution_requirements=("Retain the scPerturb MIT copyright and license notice.",),
        ),
    )
    return DatasetManifest(
        dataset_id="sciplex3-k562-24h",
        version="scperturb-1.4-review-1",
        title="sci-Plex3 K562 randomized 24-hour small-molecule endpoint",
        description=(
            "Exact K562 slice of the corrected scPerturb 1.4 sci-Plex3 H5AD: randomized "
            "small-molecule assignments at four doses, same-plate vehicle controls, and one "
            "destructive 24-hour nuclear RNA UMI endpoint."
        ),
        origin=PublicRealDataOrigin(
            repository="NCBI GEO with corrected scPerturb Zenodo harmonization",
            study_accession="GSE139944",
            publication_doi="10.1126/science.aax6234",
            release="2019-12-05",
            species=(HUMAN,),
            biological_systems=(K562,),
        ),
        sources=sources,
        slice_spec=DatasetSliceSpec(
            kind=DatasetSliceKind.CONTENT_ADDRESSED_SELECTION,
            slice_id="sciplex3-k562-exact-cell-line-slice-v1",
            selection_source_ids=(SOURCE_ID,),
            record_id_field="_index",
            selected_record_ids_uri=_repo_uri(PREP_DIR / "memberships/universe-record-ids.json"),
            selected_record_ids_sha256=universe["record_ids_sha256"],
            selected_record_count=universe["record_count"],
            selected_subject_count=universe["composite_well_count"],
            selector_id="cellstate.prepare-sciplex3-k562-k562-selector",
            selector_version="1.0.0",
            selector_sha256=prep["generator_sha256"],
            selection_stages=(
                CohortSelectionStage(
                    stage_id="select-exact-k562-source-label",
                    input_record_count=799317,
                    output_record_count=173652,
                    criterion=(
                        "obs.cell_line is exactly the UTF-8 string 'K562'; preserve every "
                        "matching source row and no others."
                    ),
                    source_ids=(SOURCE_ID,),
                ),
            ),
        ),
        use_policies=policies,
        experimental_design=ExperimentalDesign(
            units=(
                ExperimentalUnitSpec(
                    level=ExperimentalUnitLevel.CULTURE,
                    identity=CULTURE_IDENTITY,
                    source_ids=(SOURCE_ID,),
                ),
                ExperimentalUnitSpec(
                    level=ExperimentalUnitLevel.SAMPLE,
                    identity=SAMPLE_IDENTITY,
                    source_ids=(SOURCE_ID,),
                    parent_level=ExperimentalUnitLevel.CULTURE,
                ),
                ExperimentalUnitSpec(
                    level=ExperimentalUnitLevel.PLATE,
                    identity=PLATE_IDENTITY,
                    source_ids=(SOURCE_ID,),
                    parent_level=ExperimentalUnitLevel.SAMPLE,
                ),
                ExperimentalUnitSpec(
                    level=ExperimentalUnitLevel.WELL,
                    identity=WELL_IDENTITY,
                    source_ids=(SOURCE_ID,),
                    parent_level=ExperimentalUnitLevel.PLATE,
                ),
                ExperimentalUnitSpec(
                    level=ExperimentalUnitLevel.CELL,
                    identity=CELL_IDENTITY,
                    source_ids=(SOURCE_ID,),
                    parent_level=ExperimentalUnitLevel.WELL,
                ),
            ),
            sampling=SamplingDesign(
                subject_kind=SamplingSubjectKind.POPULATION,
                subject_unit=ExperimentalUnitLevel.WELL,
                subject_identity=WELL_IDENTITY,
                source_ids=(SOURCE_ID,),
                mode=SamplingMode.ENDPOINT_DESTRUCTIVE,
                linkage=SubjectLinkage.NONE,
                time_field="time",
                source_time_units="h",
            ),
            default_split_unit=ExperimentalUnitLevel.PLATE,
            biological_replicate_unit=ExperimentalUnitLevel.WELL,
            randomization_unit=ExperimentalUnitLevel.WELL,
            randomized_endpoint_contrast=RandomizedEndpointContrast(
                assignment_time_seconds=0.0,
                endpoint_time_seconds=86400.0,
                baseline_observation_present=False,
                matched_control=MatchedControlDefinition(
                    predicates=(
                        MatchedControlPredicate(
                            source_field="dose_value",
                            value_type=ControlPredicateValueType.NUMBER,
                            equals=0.0,
                        ),
                        MatchedControlPredicate(
                            source_field="perturbation",
                            value_type=ControlPredicateValueType.STRING,
                            equals="control",
                        ),
                    ),
                    stratum_identity=PLATE_IDENTITY,
                    source_ids=(SOURCE_ID,),
                ),
                source_ids=tuple(sorted((PAPER_SOURCE_ID, SOURCE_ID))),
            ),
            batch_fields=("plate", "replicate"),
        ),
        capabilities=DatasetCapabilities(
            modalities=(
                ModalitySpec(
                    modality=RNA_MODALITY,
                    source_ids=(SOURCE_ID,),
                    subject_alignment=SubjectAlignment.SAME_POPULATION,
                    alignment_group="sciplex3-k562-well-endpoint",
                    alignment_unit=ExperimentalUnitLevel.WELL,
                    raw_available=False,
                    processed_available=True,
                    destructive=True,
                    feature_identifier_namespace="Ensembl gene ID plus source feature index",
                ),
            ),
            interventions=InterventionCapability(
                source_ids=tuple(sorted((PAPER_SOURCE_ID, SOURCE_ID))),
                assignment=ManifestAssignmentMechanism.RANDOMIZED,
                kinds=compound_kinds,
                targets_recorded=False,
                doses_recorded=True,
                durations_recorded=True,
                start_stop_recorded=True,
                washout_recorded=False,
                combinations_present=False,
                assignment_probabilities_recorded=False,
                matched_controls_present=True,
                realization_evidence=RealizationEvidence.ASSIGNMENT_ONLY,
            ),
            timing=TimingCapability(
                source_ids=tuple(sorted((PAPER_SOURCE_ID, SOURCE_ID))),
                timepoints_seconds=(86400.0,),
                observation_times_recorded=True,
                intervention_times_recorded=True,
                event_ordering_recorded=True,
            ),
            lineage=LineageCapability(),
            spatial=SpatialCapability(),
            functional=FunctionalCapability(),
        ),
        claim_assessments=(snapshot_claim, intervention_claim, *negatives),
        loss_assessments=(loss,),
        metric_assessments=metrics,
        notes=(
            "The H5AD is the only executable source; GEO metadata, the primary paper, and pinned "
            "processing code are human-review evidence only.",
            "Dose labels are source-scoped and are not asserted to be normalized chemical "
            "identities.",
            "One held-out well per exact condition cannot identify latent condition-specific "
            "between-well uncertainty.",
            "The source feature axis is broader than the benchmark target; the query separately "
            "binds an ordered 2,000-feature panel selected on train only.",
            "The full-source-axis logCP10k transform recorded with the feature panel is TRAIN-only "
            "feature-selection evidence. Benchmark scoring uses a separate bound transform whose "
            "denominator is the sum of the declared 2,000 raw counts only.",
        ),
    )


def _slug(label: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", label.casefold()).strip("-")[:28]
    return f"{stem}-{hashlib.sha256(label.encode('utf-8')).hexdigest()[:12]}"


def build_query(
    labels: tuple[str, ...], support: dict[str, ContentAddressedArtifact]
) -> StateQuery:
    interventions = []
    for label in labels:
        for dose in DOSES_NM:
            interventions.append(
                InterventionSpec(
                    spec_id=f"{_slug(label)}-{dose:05d}-nm",
                    kind=_compound_kind(label),
                    target=None,
                    dose_domain=NumericDomain(minimum=float(dose), maximum=float(dose), units="nM"),
                    duration_seconds=ScalarRange(minimum=86400.0, maximum=86400.0),
                    schedule=ScheduleDomain(
                        allowed_kinds=(ScheduleKind.CONTINUOUS,),
                        administration_count=IntegerRange(minimum=1, maximum=1),
                        interval_seconds=None,
                        washout_seconds=ScalarRange(minimum=0.0, maximum=0.0),
                    ),
                    delivery_methods=("culture-medium exposure",),
                    allowed_reversibility_statuses=(ReversibilityStatus.UNKNOWN,),
                    allowed_assignment_mechanisms=(AssignmentMechanism.RANDOMIZED,),
                    assignment_unit_kind="well",
                    randomization_unit_kind="well",
                    require_randomization_unit=True,
                    require_matched_control=True,
                    realization_evidence=RealizationEvidenceRequirement(
                        allowed_statuses=(PerturbationStatus.UNKNOWN,),
                        allowed_modalities=(),
                        minimum_evidence_events=0,
                    ),
                )
            )
    interventions.sort(key=lambda item: item.spec_id)
    protocol = VersionedReference(
        reference_id="scperturb-sciplex3-corrected-endpoint-protocol",
        version="1.4",
        fingerprint=support["source_verification"].sha256,
    )
    assay = AssaySpec(
        assay_id="sciplex3-24h-scirnaseq-endpoint",
        modality=RNA_MODALITY,
        protocol_reference=protocol,
        collection=ObservationCollection(effect=CollectionEffect.TERMINAL_DESTRUCTIVE),
        purposes=(AssayPurpose.TARGET_ENDPOINT,),
    )
    return StateQuery(
        subject=SubjectSpecification(
            kind=SubjectKind.POPULATION,
            biological_system=K562,
            membership_semantics=(
                "K562 population assigned to one source well immediately before randomized "
                "active-compound or source-matched vehicle exposure at t=0"
            ),
            experimental_unit_kind="well",
            allowed_identity_bases=(IdentityBasis.EXPERIMENTAL_UNIT,),
        ),
        system_boundary=SystemBoundary.POPULATION,
        temporal_resolution_seconds=86400.0,
        prediction_horizons=(
            PredictionHorizon(
                name=HORIZON_NAME,
                duration_seconds=86400.0,
                timescale=Timescale.INTERMEDIATE,
            ),
        ),
        target_outputs=(
            OutputSpec(
                term=TARGET_TERM,
                units="raw_integer_umi_count_vector_ordered_train_derived_2000_feature_panel",
                value_schema_reference=VersionedReference(
                    reference_id="sciplex3-k562-24h-panel-count-target-value-schema",
                    version="1.1.0",
                    fingerprint=support["target_value_schema"].sha256,
                ),
                aggregation=TargetAggregation(
                    subject_kind=SubjectKind.POPULATION,
                    statistic=AggregationStatistic.DISTRIBUTION,
                    experimental_unit="well",
                ),
                endpoint=FutureAssayObservationEndpoint(
                    assay_id=assay.assay_id,
                    protocol_reference=protocol,
                ),
                missingness=TargetMissingnessSemantics(
                    policy=TargetMissingnessPolicy.REQUIRE_OBSERVED,
                    reportable_statuses=(MissingnessStatus.OBSERVED,),
                ),
                censoring=TargetCensoringSemantics(
                    policy=TargetCensoringPolicy.REJECT_CENSORED,
                    allowed_directions=(),
                ),
                supported_horizon_names=(HORIZON_NAME,),
                weight=1.0,
                functional=False,
            ),
        ),
        intervention_space=tuple(interventions),
        available_assays=(assay,),
        evidence_policy=EvidencePolicy(
            lookback_seconds=0.0,
            include_at_cutoff=True,
            allowed_modalities=(CONTEXT_MODALITY,),
            allowed_evidence_roles=(EvidenceRole.DIRECT,),
            minimum_observed_measurements=0,
        ),
        acceptance_thresholds=AcceptanceThresholds(
            maximum_ood_score=0.0,
            maximum_history_information_gain=0.0,
            minimum_calibration_coverage=0.9,
            maximum_calibration_error=0.1,
            maximum_counterfactual_uncertainty=1.0,
            maximum_decision_uncertainty=1.0,
            minimum_identifiability=1.0,
        ),
        constraints=QueryConstraints(
            maximum_intervention_combination_order=1,
            require_complete_intervention_history=True,
            require_complete_environment_history=False,
            require_complete_lineage_history=False,
            require_complete_neighborhood_history=False,
            allow_transport=False,
            maximum_total_assay_cost=None,
            assay_cost_units=None,
            maximum_assay_delay_seconds=None,
        ),
    )


def _load_preparation() -> dict[str, Any]:
    universe_path = PREP_DIR / "k562-universe.json"
    index_path = PREP_DIR / "artifact-index.json"
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    partitions = json.loads((PREP_DIR / "partitions.json").read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    source_verification = json.loads(
        (PREP_DIR / "source-verification.json").read_text(encoding="utf-8")
    )
    script_payload = (REPO_ROOT / "scripts/prepare_sciplex3_k562.py").read_bytes()
    generator_sha256 = _sha256_bytes(script_payload)
    if _sha256_bytes(index_path.read_bytes()) != (
        "3b194267337562f6bf6f8346db995e20990af89719ff2026db0246bf424c4239"
    ):
        raise RuntimeError("preparation artifact index is not the final reviewed freeze")
    if _sha256_bytes(universe_path.read_bytes()) != (
        "af77a87c1add0f818646397b1cf0b487099feae684e085968af027d525ec3399"
    ):
        raise RuntimeError("preparation universe descriptor is not the final reviewed freeze")
    if generator_sha256 != index["generator_script_sha256"]:
        raise RuntimeError("preparation artifact index does not bind the exact generator")
    if source_verification["source"]["sha256"] != (
        "603ed16c5e25401c8a7f5bb0b2b045179701017d65dcfc6aeea71722a66cd10a"
    ):
        raise RuntimeError("unexpected sci-Plex3 source identity")
    for artifact in index["artifacts"]:
        payload = (PREP_DIR / artifact["relative_path"]).read_bytes()
        if (len(payload), _sha256_bytes(payload)) != (
            artifact["byte_count"],
            artifact["sha256"],
        ):
            raise RuntimeError(
                f"preparation artifact {artifact['relative_path']} does not match the final index"
            )

    partitions_by_role = {item["partition_role"]: item for item in partitions["partitions"]}
    if tuple(partitions_by_role) != ROLES:
        raise RuntimeError("preparation partitions do not use the frozen role order")
    for role in ("universe", *ROLES):
        descriptor = universe if role == "universe" else partitions_by_role[role]
        membership_artifacts = descriptor["membership_artifacts"]
        if set(membership_artifacts) != {
            "plate_ids",
            "record_ids",
            "record_to_well",
            "well_ids",
            "well_to_condition",
        }:
            raise RuntimeError(f"{role} does not declare the complete frozen membership set")
        for artifact in membership_artifacts.values():
            path = PREP_DIR / artifact["relative_path"]
            payload = path.read_bytes()
            if (len(payload), _sha256_bytes(payload)) != (
                artifact["byte_count"],
                artifact["sha256"],
            ):
                raise RuntimeError(
                    f"materialized membership {artifact['relative_path']} does not match its ledger"
                )
    groups = json.loads((PREP_DIR / "well-groups.json").read_text(encoding="utf-8"))
    labels = tuple(
        sorted(
            {
                item["normalized_perturbation_label"]
                for item in groups["groups"]
                if not item["is_vehicle_control"]
            }
        )
    )
    if len(labels) != 188 or "control" in labels:
        raise RuntimeError("frozen source label domain must contain exactly 188 treatments")
    return {
        "generator_sha256": generator_sha256,
        "groups": groups,
        "index": index,
        "labels": labels,
        "partitions": partitions,
        "partitions_by_role": partitions_by_role,
        "source_verification": source_verification,
        "universe": universe,
    }


def _write_support_specs(prep: dict[str, Any]) -> dict[str, ContentAddressedArtifact]:
    labels = prep["labels"]
    groups = prep["groups"]["groups"]
    source_by_normalized: dict[str, set[str]] = {}
    for group in groups:
        source_by_normalized.setdefault(group["normalized_perturbation_label"], set()).add(
            group["source_perturbation_label"]
        )
    label_mapping_path = PREP_DIR / "mappings" / "source-label-to-normalized-label.json"
    label_mapping = json.loads(label_mapping_path.read_text(encoding="utf-8"))
    expected_label_mapping = sorted(
        [source_label, normalized_label]
        for normalized_label, source_labels in source_by_normalized.items()
        for source_label in source_labels
    )
    if label_mapping != expected_label_mapping:
        raise RuntimeError("source-label normalization does not match the frozen well groups")
    label_mapping_artifact = _artifact_for_file(
        "sciplex3-source-label-to-normalized-label",
        label_mapping_path,
    )
    feature_panel_path = PREP_DIR / "feature-panel.json"
    feature_panel = json.loads(feature_panel_path.read_text(encoding="utf-8"))
    feature_panel_artifact = _artifact_for_file(
        "sciplex3-train-feature-panel",
        feature_panel_path,
    )
    feature_selection_transform = feature_panel["transformation"]
    if (
        feature_panel["feature_count"] != 2000
        or feature_selection_transform["library_size"]
        != "sum of all source-axis UMI counts for that record"
    ):
        raise RuntimeError("frozen feature-selection transform metadata has changed")
    _write_pretty(
        SUPPORT_DIR / "scoring-transform.json",
        {
            "artifact_schema": "sciplex3-k562-panel-only-scoring-transform",
            "artifact_schema_version": "1.0.0",
            "application": {
                "applies_to": ["observed_sample", "predicted_posterior_sample"],
                "application_unit": "one recovered-nucleus ordered count vector",
                "apply_before": [
                    "marginal_crps",
                    "joint_energy_projection",
                    "vehicle_relative_pseudobulk_effect",
                    "four_dose_profile",
                    "marginal_interval_coverage",
                    "marginal_interval_width",
                ],
                "symmetry_requirement": (
                    "apply the identical transform independently to every observed and "
                    "predicted sample"
                ),
            },
            "declared_panel": {
                "feature_count": feature_panel["feature_count"],
                "ordered_feature_keys_encoded_byte_count": feature_panel[
                    "ordered_feature_keys_encoded_byte_count"
                ],
                "ordered_feature_keys_encoding": feature_panel["ordered_feature_keys_encoding"],
                "ordered_feature_keys_sha256": feature_panel["ordered_feature_keys_sha256"],
                "panel_artifact": feature_panel_artifact.model_dump(mode="json"),
            },
            "input_contract": {
                "coordinate_order": "exact declared_panel ordered feature-key order",
                "coordinate_value_domain": "finite nonnegative integer raw UMI count",
                "external_denominator_inputs": [],
                "feature_count": feature_panel["feature_count"],
                "full_source_axis_input_allowed": False,
                "raw_value_schema": (
                    "one exact-length ordered vector of raw integer UMI counts on the declared "
                    "2,000-feature panel"
                ),
            },
            "output_contract": {
                "coordinate_order": "unchanged from the exact declared panel order",
                "feature_count": feature_panel["feature_count"],
                "value_domain": "finite nonnegative real panel_logCP10k value",
            },
            "scoring_transform": {
                "denominator_definition": (
                    "panel_total = sum(count_i for i in the exact ordered 2,000-feature panel)"
                ),
                "denominator_scope": "declared_ordered_2000_feature_panel_only",
                "formula": "log1p(10000 * count_i / panel_total)",
                "log_base": "e",
                "scale": 10000,
                "transformation_id": "panel-only-natural-log-cp10k",
                "transformation_version": "1.0.0",
            },
            "train_feature_selection_transform": {
                "full_source_axis_denominator_used": True,
                "purpose": "TRAIN-only feature ranking and selection evidence",
                "source_metadata": feature_selection_transform,
                "used_for_benchmark_scoring": False,
            },
            "validation_policy": {
                "failure_action": (
                    "fail the metric evaluation and therefore block benchmark admission; do not "
                    "drop, exclude, impute, renormalize, clip, or substitute the sample"
                ),
                "fatal_conditions": [
                    "coordinate_order_missing_or_not_exactly_bound",
                    "nonfinite_coordinate",
                    "noninteger_coordinate",
                    "negative_coordinate",
                    "panel_total_less_than_or_equal_to_zero",
                    "vector_length_not_exactly_2000",
                ],
                "zero_panel_total_policy": "error_fail_evaluation_no_exclusion_or_imputation",
            },
        },
    )
    scoring_transform_artifact = _artifact_for_file(
        "sciplex3-panel-only-scoring-transform",
        SUPPORT_DIR / "scoring-transform.json",
    )
    _write_pretty(
        SUPPORT_DIR / "target-value-schema.json",
        {
            "artifact_schema": "sciplex3-k562-panel-count-target-value-schema",
            "artifact_schema_version": "1.1.0",
            "feature_axis": {
                "feature_count": feature_panel["feature_count"],
                "ordered_feature_keys_encoded_byte_count": feature_panel[
                    "ordered_feature_keys_encoded_byte_count"
                ],
                "ordered_feature_keys_encoding": feature_panel["ordered_feature_keys_encoding"],
                "ordered_feature_keys_sha256": feature_panel["ordered_feature_keys_sha256"],
                "panel_artifact": feature_panel_artifact.model_dump(mode="json"),
            },
            "raw_value_contract": {
                "coordinate_value_domain": "finite nonnegative integer raw UMI count",
                "feature_count": feature_panel["feature_count"],
                "feature_order": "exactly the bound ordered feature-key array",
                "serialization_or_adapter_requirement": (
                    "supply and verify the exact ordered-feature SHA-256 before evaluation"
                ),
                "units": ("raw_integer_umi_count_vector_ordered_train_derived_2000_feature_panel"),
            },
            "scoring_input_sufficiency": {
                "external_inputs_required": [],
                "full_source_axis_counts_required": False,
                "statement": (
                    "The declared raw 2,000-coordinate target contains every value needed to "
                    "compute the scoring denominator."
                ),
            },
            "scoring_transform_artifact": scoring_transform_artifact.model_dump(mode="json"),
            "validation_policy": {
                "failure_action": "error and block evaluation with no exclusion or imputation",
                "fatal_conditions": [
                    "coordinate_order_missing_or_mismatch",
                    "nonfinite_coordinate",
                    "noninteger_coordinate",
                    "negative_coordinate",
                    "panel_total_less_than_or_equal_to_zero",
                    "vector_length_not_exactly_2000",
                ],
            },
            "value_schema_id": "sciplex3-k562-24h-panel-count-target-value-schema",
            "value_schema_version": "1.1.0",
        },
    )
    target_value_schema_artifact = _artifact_for_file(
        "sciplex3-panel-count-target-value-schema",
        SUPPORT_DIR / "target-value-schema.json",
    )
    action_entries = []
    for label in labels:
        for dose in DOSES_NM:
            action_entries.append(
                {
                    "dose_nm": dose,
                    "duration_seconds": 86400,
                    "intervention_kind_key": _compound_kind(label).key,
                    "query_spec_id": f"{_slug(label)}-{dose:05d}-nm",
                    "normalized_perturbation_label": label,
                    "source_perturbation_labels": sorted(source_by_normalized[label]),
                    "source_scoped_condition_id": f"source-label:{label}@{dose}nM",
                }
            )
    action_entries.sort(key=lambda item: item["query_spec_id"])
    controls_by_plate = {
        plate: tuple(
            sorted(
                group["composite_well_id"]
                for group in groups
                if group["plate"] == plate and group["is_vehicle_control"]
            )
        )
        for plate in sorted({group["plate"] for group in groups})
    }
    if any(len(control_ids) != 2 for control_ids in controls_by_plate.values()):
        raise RuntimeError("every physical plate must have exactly two no-action controls")
    treated_control_rows = tuple(
        (
            group["composite_well_id"],
            list(controls_by_plate[group["plate"]]),
        )
        for group in sorted(groups, key=lambda item: item["composite_well_id"])
        if not group["is_vehicle_control"]
    )
    _write_canonical(
        SUPPORT_DIR / "treated-well-to-matched-controls.json",
        treated_control_rows,
    )
    _write_canonical(
        SUPPORT_DIR / "well-to-compound.json",
        sorted(
            [
                group["composite_well_id"],
                group["normalized_perturbation_label"],
            ]
            for group in groups
        ),
    )
    _write_canonical(
        SUPPORT_DIR / "well-to-plate.json",
        sorted([group["composite_well_id"], group["plate"]] for group in groups),
    )
    for plate in sorted(controls_by_plate):
        group = next(item for item in groups if item["plate"] == plate)
        context = {
            "biological_system_key": K562.key,
            "context_schema": "sciplex3-k562-static-plate-context-v1",
            "endpoint_protocol": "corrected-scperturb-1.4-sciplex3-24h",
            "plate": plate,
            "population_boundary": (
                "K562 population assigned to the source well immediately before exposure at t=0"
            ),
            "replicate": group["replicate"],
            "source_sha256": "603ed16c5e25401c8a7f5bb0b2b045179701017d65dcfc6aeea71722a66cd10a",
            "vehicle_background": (
                "Source-protocol vehicle background is static context, not an active-compound "
                "action; controls have zero modeled active compound."
            ),
        }
        _write_canonical(SUPPORT_DIR / "contexts" / f"{plate}.json", context)
    _write_pretty(
        SUPPORT_DIR / "action-domain-mapping.json",
        {
            "artifact_schema": "sciplex3-k562-query-action-domain",
            "artifact_schema_version": "1.0.0",
            "combination_order": 1,
            "dose_values_nm": list(DOSES_NM),
            "entries": action_entries,
            "entry_count": len(action_entries),
            "identity_semantics": (
                "Each intervention kind denotes administration of one exact source-scoped "
                "active-compound label beyond the source-protocol vehicle background; biological "
                "target is null and no chemical ontology equivalence or target engagement is "
                "asserted."
            ),
            "source_label_normalization": {
                "artifact": label_mapping_artifact.model_dump(mode="json"),
                "operation": "trim leading and trailing Unicode whitespace only",
                "ontology_or_chemical_mapping_asserted": False,
            },
        },
    )
    _write_pretty(
        SUPPORT_DIR / "target-semantics.json",
        {
            "artifact_schema": "sciplex3-k562-target-semantics",
            "artifact_schema_version": "1.0.0",
            "assay_endpoint": "destructive 24-hour sci-RNA-seq endpoint",
            "feature_axis": {
                "count": feature_panel["feature_count"],
                "ordered_feature_keys_encoded_byte_count": feature_panel[
                    "ordered_feature_keys_encoded_byte_count"
                ],
                "ordered_feature_keys_encoding": feature_panel["ordered_feature_keys_encoding"],
                "ordered_feature_keys_sha256": feature_panel["ordered_feature_keys_sha256"],
                "panel_artifact": feature_panel_artifact.model_dump(mode="json"),
                "selection_partition_role": "train",
            },
            "population_boundary": "recovered K562 nuclei from one independently treated well",
            "reference_control": (
                "same-plate wells with zero active compound under the common source-matched "
                "vehicle background; no-action means no modeled active-compound action"
            ),
            "target_key": TARGET_TERM.key,
            "target_units": (
                "raw_integer_umi_count_vector_ordered_train_derived_2000_feature_panel"
            ),
            "target_value_schema": target_value_schema_artifact.model_dump(mode="json"),
            "scoring_semantics": {
                "full_source_axis_counts_required": False,
                "panel_total": (
                    "sum of the exact 2,000 declared ordered raw-count coordinates for each "
                    "observed or predicted sample"
                ),
                "scoring_transform_artifact": scoring_transform_artifact.model_dump(mode="json"),
                "zero_panel_total_policy": ("error_fail_evaluation_no_exclusion_or_imputation"),
            },
            "truth_conditions": {
                "is_cell_trajectory": False,
                "is_latent_living_cell_state": False,
                "is_total_population_size": False,
                "is_universal_cell_state_feature_space": False,
                "is_viability": False,
                "values_are_integer_umi_counts": True,
            },
        },
    )
    _write_pretty(
        SUPPORT_DIR / "metric-suite-spec.json",
        {
            "artifact_schema": "sciplex3-k562-frozen-metric-suite",
            "artifact_schema_version": "1.0.0",
            "acceptance": {
                "coverage_error_upper_confidence_bound_maximum": 0.03,
                "coverage_nominal_probabilities": [0.5, 0.8, 0.95],
                "crps": (
                    "paired compound-block superiority to every mandatory probabilistic baseline"
                ),
                "joint_energy_and_effect_rmse": (
                    "both within a 2% paired relative noninferiority margin of the best applicable "
                    "mandatory baseline and at least one strictly superior"
                ),
            },
            "execution_status": "definition_frozen_not_run",
            "feature_panel": {
                "path": "benchmarks/artifacts/sciplex3-k562-24h-v1/feature-panel.json",
                "sha256": _sha256_bytes((PREP_DIR / "feature-panel.json").read_bytes()),
                "fit_partition": "train",
                "heldout_count_rows_accessed": 0,
                "full_axis_logcp10k_purpose": "TRAIN-only feature ranking and selection",
                "full_axis_logcp10k_used_for_scoring": False,
            },
            "target_value_schema": target_value_schema_artifact.model_dump(mode="json"),
            "scoring_transform": scoring_transform_artifact.model_dump(mode="json"),
            "scoring_transform_rule": (
                "For each observed or predicted sample, panel_total is the sum of exactly the "
                "declared ordered 2,000 raw counts and each coordinate becomes "
                "log1p(10000*count_i/panel_total). No full-axis denominator is read. A wrong "
                "length/order, nonfinite/noninteger/negative count, or panel_total<=0 is a fatal "
                "evaluation error with no exclusion or imputation."
            ),
            "metrics": [
                {
                    "metric_id": "sciplex3.marginal-crps-logcp10k",
                    "formula": (
                        "mean over wells and train-selected features of E|X-Y|-0.5E|X-X'| after "
                        "panel-only log1p(10000*count_i/panel_total_2000)"
                    ),
                    "primary": True,
                },
                {
                    "metric_id": "sciplex3.joint-energy-train-pca",
                    "formula": (
                        "mean over wells of E||X-Y||_2-0.5E||X-X'||_2 in a projection fit on "
                        "train only after the panel-only scoring transform"
                    ),
                    "primary": True,
                },
                {
                    "metric_id": "sciplex3.vehicle-relative-pseudobulk-rmse",
                    "formula": (
                        "RMSE of predicted versus observed treated-minus-same-plate-vehicle mean "
                        "panel-logCP10k effects"
                    ),
                    "primary": True,
                },
                {
                    "metric_id": "sciplex3.marginal-coverage-error-p50",
                    "formula": (
                        "absolute empirical marginal coverage error from nominal 0.50 on the "
                        "panel-logCP10k untouched-test values"
                    ),
                    "primary": True,
                },
                {
                    "metric_id": "sciplex3.marginal-coverage-error-p80",
                    "formula": (
                        "absolute empirical marginal coverage error from nominal 0.80 on the "
                        "panel-logCP10k untouched-test values"
                    ),
                    "primary": True,
                },
                {
                    "metric_id": "sciplex3.marginal-coverage-error-p95",
                    "formula": (
                        "absolute empirical marginal coverage error from nominal 0.95 on the "
                        "panel-logCP10k untouched-test values"
                    ),
                    "primary": True,
                },
                {
                    "metric_id": "sciplex3.marginal-interval-width-p50",
                    "formula": (
                        "mean central 50% panel-logCP10k predictive interval width on untouched "
                        "test wells"
                    ),
                    "primary": False,
                },
                {
                    "metric_id": "sciplex3.marginal-interval-width-p80",
                    "formula": (
                        "mean central 80% panel-logCP10k predictive interval width on untouched "
                        "test wells"
                    ),
                    "primary": False,
                },
                {
                    "metric_id": "sciplex3.marginal-interval-width-p95",
                    "formula": (
                        "mean central 95% panel-logCP10k predictive interval width on untouched "
                        "test wells"
                    ),
                    "primary": False,
                },
                {
                    "metric_id": "sciplex3.four-dose-profile-diagnostic",
                    "formula": (
                        "equal-compound mean RMSE over the four exact-dose vehicle-relative "
                        "panel-logCP10k pseudobulk profiles"
                    ),
                    "primary": False,
                },
            ],
            "missingness": (
                "Every expected well is required; an absent target is an error and never silently "
                "dropped."
            ),
            "predictive_interval_calibration": (
                "Any interval-calibration parameters are fit only on the calibration partition; "
                "coverage error and width are reported on untouched test wells."
            ),
            "uncertainty": (
                "Multiway bootstrap by compound and whole protected plate; cells are never "
                "independent resampling units."
            ),
        },
    )
    _write_pretty(
        SUPPORT_DIR / "baseline-suite-spec.json",
        {
            "artifact_schema": "sciplex3-k562-frozen-baseline-suite",
            "artifact_schema_version": "1.0.0",
            "baselines": [
                "exact-condition-negative-binomial",
                "exact-condition-rep1-empirical-resampling",
                "hierarchical-well-negative-binomial",
                "low-rank-compound-dose-response",
                "matched-vehicle-resampling",
                "nearest-supported-dose",
                "persistence",
                "temporal-state-space",
            ],
            "execution_status": "not_run",
            "inapplicability": {
                "persistence": (
                    "requires a pre-cutoff target-modality observation; frozen query has none"
                ),
                "temporal-state-space": (
                    "also requires at least two future horizons; frozen query has one"
                ),
            },
            "mandatory_probabilistic_baselines": list(MANDATORY_PROBABILISTIC_BASELINE_IDS),
            "scoring_contract": {
                "candidate_and_baseline_symmetry": (
                    "candidate and every applicable baseline emit raw ordered 2,000-panel count "
                    "samples and are scored with the identical panel-only transform"
                ),
                "scoring_transform_artifact": scoring_transform_artifact.model_dump(mode="json"),
                "zero_panel_total_policy": ("error_fail_evaluation_no_exclusion_or_imputation"),
            },
            "rule": (
                "A missing or crashed applicable baseline blocks admission and cannot be relabeled "
                "not applicable."
            ),
            "secondary_applicable_baselines": ["nearest-supported-dose"],
        },
    )
    _write_pretty(
        SUPPORT_DIR / "leakage-audit-evidence.json",
        {
            "artifact_schema": "sciplex3-k562-definition-leakage-review",
            "artifact_schema_version": "1.0.0",
            "checks_from_preparation": prep["partitions"]["cross_partition_checks"],
            "feature_fit": {
                "count_accessed_partition_roles": ["train"],
                "full_source_axis_logcp10k_purpose": "TRAIN-only feature ranking and selection",
                "heldout_count_rows_accessed": 0,
            },
            "scoring_transform": {
                "fitted_state": "none",
                "full_source_axis_counts_required": False,
                "input_scope": "declared ordered 2,000-feature raw count vector only",
                "scoring_transform_artifact": scoring_transform_artifact.model_dump(mode="json"),
                "zero_panel_total_policy": ("error_fail_evaluation_no_exclusion_or_imputation"),
            },
            "limitations": [
                "No model or applicable baseline has been executed.",
                "Source-duplicate detection beyond exact source row IDs remains unassessed.",
                "All assessment bindings reuse one explicitly identified physical dataset and "
                "one four-way physical split.",
            ],
            "temporal_cutoff": {
                "cutoff_seconds": 0,
                "pre_cutoff_target_observations": 0,
                "target_endpoint_seconds": 86400,
            },
        },
    )
    artifacts = {
        "action_domain": _artifact_for_file(
            "sciplex3-query-action-domain", SUPPORT_DIR / "action-domain-mapping.json"
        ),
        "baseline_suite": _artifact_for_file(
            "sciplex3-frozen-baseline-suite", SUPPORT_DIR / "baseline-suite-spec.json"
        ),
        "feature_panel": feature_panel_artifact,
        "leakage_evidence": _artifact_for_file(
            "sciplex3-definition-leakage-review",
            SUPPORT_DIR / "leakage-audit-evidence.json",
        ),
        "metric_suite": _artifact_for_file(
            "sciplex3-frozen-metric-suite", SUPPORT_DIR / "metric-suite-spec.json"
        ),
        "scoring_transform": scoring_transform_artifact,
        "source_verification": _artifact_for_file(
            "sciplex3-source-verification", PREP_DIR / "source-verification.json"
        ),
        "target_semantics": _artifact_for_file(
            "sciplex3-target-semantics", SUPPORT_DIR / "target-semantics.json"
        ),
        "target_value_schema": target_value_schema_artifact,
        "well_groups": _artifact_for_file(
            "sciplex3-well-descendant-groups", PREP_DIR / "well-groups.json"
        ),
        "matched_controls": _artifact_for_file(
            "sciplex3-treated-well-to-matched-controls",
            SUPPORT_DIR / "treated-well-to-matched-controls.json",
        ),
        "source_label_mapping": label_mapping_artifact,
        "well_to_compound": _artifact_for_file(
            "sciplex3-well-to-source-scoped-compound",
            SUPPORT_DIR / "well-to-compound.json",
        ),
        "well_to_condition": _membership_artifact("universe", "well-to-condition"),
        "well_to_plate": _artifact_for_file(
            "sciplex3-well-to-plate",
            SUPPORT_DIR / "well-to-plate.json",
        ),
    }
    return artifacts


def _assessment_reference(manifest: DatasetManifest, assessment: Any) -> DatasetAssessmentReference:
    return DatasetAssessmentReference(
        dataset_manifest_fingerprint=manifest.fingerprint,
        assessment_id=assessment.assessment_id,
        assessment_fingerprint=assessment.fingerprint,
    )


def _target_mapping(
    query: StateQuery, support: dict[str, ContentAddressedArtifact]
) -> EvidenceTargetMapping:
    output = query.target_outputs[0]
    return EvidenceTargetMapping(
        target_output_key=output.term.key,
        target_output_fingerprint=canonical_fingerprint(output.model_dump(mode="json")),
        target_units=output.units,
        target_aggregation=output.aggregation,
        aggregation_unit=_unit(ExperimentalUnitLevel.WELL),
        assessment_modalities=(RNA_MODALITY.key,),
        semantics_artifact=support["target_semantics"],
    )


def _intervention_mappings(
    query: StateQuery, support: dict[str, ContentAddressedArtifact]
) -> tuple[EvidenceInterventionMapping, ...]:
    return tuple(
        EvidenceInterventionMapping(
            intervention_spec_id=spec.spec_id,
            intervention_spec_fingerprint=canonical_fingerprint(spec.model_dump(mode="json")),
            assessment_intervention_kind_key=spec.kind.key,
            domain_mapping_artifact=support["action_domain"],
        )
        for spec in sorted(query.intervention_space, key=lambda item: item.spec_id)
    )


def build_evidence_bindings(
    manifest: DatasetManifest,
    query: StateQuery,
    manifest_artifact: ContentAddressedArtifact,
    support: dict[str, ContentAddressedArtifact],
) -> tuple[BenchmarkEvidenceBinding, ...]:
    target_mappings = (_target_mapping(query, support),)
    intervention_mappings = _intervention_mappings(query, support)
    loss = manifest.loss_assessments[0]
    rows: list[tuple[str, Any, AssessmentKind, Any]] = [
        (
            "loss-intervention-effect",
            loss,
            AssessmentKind.LOSS,
            LossAssessmentIdentity(
                assessment_kind=AssessmentKind.LOSS,
                loss_kind=loss.loss_kind,
            ),
        )
    ]
    for index, assessment in enumerate(
        sorted(manifest.metric_assessments, key=lambda item: item.metric_id), start=1
    ):
        rows.append(
            (
                f"metric-{index:02d}-{assessment.metric_id}",
                assessment,
                AssessmentKind.METRIC,
                MetricAssessmentIdentity(
                    assessment_kind=AssessmentKind.METRIC,
                    metric_id=assessment.metric_id,
                    metric_family=assessment.metric_family,
                    partition_purpose=assessment.partition_purpose,
                ),
            )
        )
    bindings = []
    for binding_id, assessment, kind, identity in rows:
        bindings.append(
            BenchmarkEvidenceBinding(
                binding_id=binding_id,
                physical_dataset_binding_id=PHYSICAL_DATASET_BINDING_ID,
                dataset_id=manifest.dataset_id,
                dataset_version=manifest.version,
                manifest_artifact=manifest_artifact,
                manifest_fingerprint=manifest.fingerprint,
                assessment_reference=_assessment_reference(manifest, assessment),
                assessment_kind=kind,
                assessment_identity=identity,
                scope_binding=EvidenceScopeBinding(
                    assessment_scope_fingerprint=assessment.scope.fingerprint,
                    target_mappings=target_mappings,
                    intervention_mappings=intervention_mappings,
                    horizon_names=(HORIZON_NAME,),
                    scientific_claims=(ScientificClaim.INTERVENTION_EFFECT,),
                ),
                required_split_unit=_unit(ExperimentalUnitLevel.PLATE),
            )
        )
    return tuple(sorted(bindings, key=lambda item: item.binding_id))


def _id_membership(role: str, name: str, count: int) -> CanonicalIdMembership:
    return CanonicalIdMembership(
        ids_artifact=_membership_artifact(role, name),
        id_count=count,
    )


def _protected_memberships(
    role: str,
    *,
    plate_count: int,
    well_count: int,
) -> tuple[ProtectedGroupMembership, ...]:
    members = (
        ProtectedGroupMembership(
            unit=_unit(ExperimentalUnitLevel.PLATE),
            membership=_id_membership(role, "plate-ids", plate_count),
        ),
        ProtectedGroupMembership(
            unit=_unit(ExperimentalUnitLevel.WELL),
            membership=_id_membership(role, "well-ids", well_count),
        ),
    )
    return tuple(sorted(members, key=lambda item: item.unit.key))


def _explicit_membership(role: str, partition: dict[str, Any]) -> ExplicitPartitionMembership:
    plate_count = len(partition["selector"]["plate"])
    return ExplicitPartitionMembership(
        assignment_unit=_unit(ExperimentalUnitLevel.PLATE),
        assignment_unit_ids=_id_membership(role, "plate-ids", plate_count),
        record_unit=_unit(ExperimentalUnitLevel.CELL),
        record_ids=_id_membership(role, "record-ids", partition["record_count"]),
        descendant_closure_artifact=_membership_artifact(role, "record-to-well"),
        protected_group_memberships=_protected_memberships(
            role,
            plate_count=plate_count,
            well_count=partition["well_count"],
        ),
    )


def build_split_plan(
    manifest: DatasetManifest,
    query: StateQuery,
    evidence_bindings: tuple[BenchmarkEvidenceBinding, ...],
    prep: dict[str, Any],
) -> BenchmarkSplitPlan:
    universe = prep["universe"]
    partition_by_role = prep["partitions_by_role"]
    if {binding.physical_dataset_binding_id for binding in evidence_bindings} != {
        PHYSICAL_DATASET_BINDING_ID
    }:
        raise RuntimeError("all sci-Plex assessment views must share one physical source")
    universe_membership = PartitionUniverse(
        physical_dataset_binding_id=PHYSICAL_DATASET_BINDING_ID,
        slice_fingerprint=manifest.slice_spec.fingerprint,
        assignment_unit=_unit(ExperimentalUnitLevel.PLATE),
        assignment_unit_ids=_id_membership("universe", "plate-ids", 16),
        record_unit=_unit(ExperimentalUnitLevel.CELL),
        record_ids=_id_membership("universe", "record-ids", universe["record_count"]),
        descendant_closure_artifact=_membership_artifact("universe", "record-to-well"),
    )
    ancestry = tuple(
        _unit(level)
        for level in (
            ExperimentalUnitLevel.CULTURE,
            ExperimentalUnitLevel.SAMPLE,
            ExperimentalUnitLevel.PLATE,
            ExperimentalUnitLevel.WELL,
            ExperimentalUnitLevel.CELL,
        )
    )
    protected_groups = tuple(
        sorted(
            (
                ProtectedGroupBinding(
                    unit=_unit(ExperimentalUnitLevel.PLATE),
                    reasons=(
                        ProtectedGroupReason.DEFAULT_SPLIT,
                        ProtectedGroupReason.OBJECTIVE_REQUIRED_SPLIT,
                        ProtectedGroupReason.SPLIT_ASSIGNMENT,
                    ),
                ),
                ProtectedGroupBinding(
                    unit=_unit(ExperimentalUnitLevel.WELL),
                    reasons=(
                        ProtectedGroupReason.BIOLOGICAL_REPLICATE,
                        ProtectedGroupReason.METRIC_EVALUATION,
                        ProtectedGroupReason.RANDOMIZATION,
                        ProtectedGroupReason.SAMPLING_SUBJECT,
                    ),
                ),
            ),
            key=lambda item: item.unit.key,
        )
    )
    role_map = {
        "train": ("p1-train", BenchmarkPartitionRole.TRAIN),
        "calibration": ("p2-calibration", BenchmarkPartitionRole.CALIBRATION),
        "model_selection_validation": (
            "p3-model-selection-validation",
            BenchmarkPartitionRole.MODEL_SELECTION_VALIDATION,
        ),
        "untouched_test": ("p4-untouched-test", BenchmarkPartitionRole.UNTOUCHED_TEST),
    }
    closure = ProtectedGroupClosure(
        physical_dataset_binding_id=PHYSICAL_DATASET_BINDING_ID,
        unit_ancestry=ancestry,
        record_unit=_unit(ExperimentalUnitLevel.CELL),
        assignment_unit=_unit(ExperimentalUnitLevel.PLATE),
        protected_groups=protected_groups,
    )
    partitions = []
    for role in ROLES:
        partition_id, benchmark_role = role_map[role]
        partitions.append(
            BenchmarkPartition(
                partition_id=partition_id,
                role=benchmark_role,
                physical_dataset_binding_id=PHYSICAL_DATASET_BINDING_ID,
                membership=_explicit_membership(role, partition_by_role[role]),
            )
        )
    return BenchmarkSplitPlan(
        split_id="sciplex3-k562-whole-plate-replicate-transfer",
        split_version="1.0.0",
        query_fingerprint=query.fingerprint,
        universes=(universe_membership,),
        protected_group_closures=(closure,),
        partitions=tuple(sorted(partitions, key=lambda item: item.partition_id)),
    )


def _specification_only(
    specification_artifact: ContentAddressedArtifact,
    *,
    component: str,
) -> SpecificationOnlyImplementationBinding:
    return SpecificationOnlyImplementationBinding(
        specification_artifact=specification_artifact,
        blockers=tuple(
            sorted(
                (
                    f"Executable {component} code has not been implemented or versioned.",
                    f"Golden fixtures for {component} have not been authored and passed.",
                )
            )
        ),
    )


def _metric_uncertainty(
    metric_id: str,
    support: dict[str, ContentAddressedArtifact],
) -> MetricUncertaintySpec:
    method = _specification_only(
        support["metric_suite"],
        component=f"{metric_id} multiway-bootstrap uncertainty",
    )
    return MetricUncertaintySpec(
        method=method,
        resampling_scheme=MetricResamplingScheme.MULTIWAY_CLUSTERED,
        dependence_units=(
            MetricDependenceUnit(
                dependence_id="compound",
                kind=MetricDependenceKind.INTERVENTION_CONDITION,
                identity=COMPOUND_IDENTITY,
                record_to_group_artifact=support["well_to_compound"],
            ),
            MetricDependenceUnit(
                dependence_id="plate",
                kind=MetricDependenceKind.EXPERIMENTAL_UNIT,
                identity=PLATE_IDENTITY,
                record_to_group_artifact=support["well_to_plate"],
                experimental_unit=_unit(ExperimentalUnitLevel.PLATE),
            ),
        ),
        confidence_level=0.95,
        resample_count=2000,
    )


def build_metrics(
    query: StateQuery,
    evidence_bindings: tuple[BenchmarkEvidenceBinding, ...],
    split_plan: BenchmarkSplitPlan,
    support: dict[str, ContentAddressedArtifact],
) -> tuple[BenchmarkMetricDefinition, ...]:
    binding_by_metric_id = {
        binding.assessment_identity.metric_id: binding
        for binding in evidence_bindings
        if isinstance(binding.assessment_identity, MetricAssessmentIdentity)
    }
    test_partition_id = next(
        partition.partition_id
        for partition in split_plan.partitions
        if partition.role is BenchmarkPartitionRole.UNTOUCHED_TEST
    )
    specs = (
        {
            "metric_id": "sciplex3.four-dose-profile-diagnostic",
            "family": MetricFamily.INTERVENTION_EFFECT,
            "formula": (
                "equal_compound_mean(rmse_dose(predicted_vehicle_relative_panel_logcp10k_"
                "pseudobulk,observed_vehicle_relative_panel_logcp10k_pseudobulk))"
            ),
            "units": "panel_logCP10k_effect_rmse",
            "weighting": MetricWeightingScheme.EQUAL_GROUP_THEN_EQUAL_EVALUATION_UNIT,
            "parameters": (
                BenchmarkParameter(name="dose_count", value=4),
                BenchmarkParameter(name="secondary_diagnostic", value=True),
            ),
        },
        {
            "metric_id": "sciplex3.joint-energy-train-pca",
            "family": MetricFamily.PREDICTIVE_PROPER_SCORE,
            "formula": (
                "mean_well(E_norm2(X,Y)-0.5*E_norm2(X,X_prime))_after_panel_only_logcp10k_"
                "in_train_only_projection"
            ),
            "units": "projected_panel_logCP10k",
            "weighting": MetricWeightingScheme.EQUAL_EVALUATION_UNIT,
            "parameters": (
                BenchmarkParameter(name="projection_fit_partition", value="train"),
                BenchmarkParameter(name="projection_rank", value=50),
            ),
        },
        {
            "metric_id": "sciplex3.marginal-coverage-error-p50",
            "family": MetricFamily.CALIBRATION,
            "formula": (
                "abs(empirical_marginal_coverage(panel_logcp10k_central_50_percent_interval)-0.50)"
            ),
            "units": "absolute_coverage_error",
            "weighting": MetricWeightingScheme.EQUAL_EVALUATION_UNIT,
            "parameters": (
                BenchmarkParameter(name="calibration_fit_partition", value="calibration"),
                BenchmarkParameter(name="nominal_coverage", value=0.5),
            ),
        },
        {
            "metric_id": "sciplex3.marginal-coverage-error-p80",
            "family": MetricFamily.CALIBRATION,
            "formula": (
                "abs(empirical_marginal_coverage(panel_logcp10k_central_80_percent_interval)-0.80)"
            ),
            "units": "absolute_coverage_error",
            "weighting": MetricWeightingScheme.EQUAL_EVALUATION_UNIT,
            "parameters": (
                BenchmarkParameter(name="calibration_fit_partition", value="calibration"),
                BenchmarkParameter(name="nominal_coverage", value=0.8),
            ),
        },
        {
            "metric_id": "sciplex3.marginal-coverage-error-p95",
            "family": MetricFamily.CALIBRATION,
            "formula": (
                "abs(empirical_marginal_coverage(panel_logcp10k_central_95_percent_interval)-0.95)"
            ),
            "units": "absolute_coverage_error",
            "weighting": MetricWeightingScheme.EQUAL_EVALUATION_UNIT,
            "parameters": (
                BenchmarkParameter(name="calibration_fit_partition", value="calibration"),
                BenchmarkParameter(name="nominal_coverage", value=0.95),
            ),
        },
        {
            "metric_id": "sciplex3.marginal-interval-width-p50",
            "family": MetricFamily.CALIBRATION,
            "formula": (
                "mean_well_feature(panel_logcp10k_central_50_percent_predictive_interval_width)"
            ),
            "units": "panel_logCP10k_interval_width",
            "weighting": MetricWeightingScheme.EQUAL_EVALUATION_UNIT,
            "parameters": (
                BenchmarkParameter(name="calibration_fit_partition", value="calibration"),
                BenchmarkParameter(name="diagnostic", value=True),
                BenchmarkParameter(name="nominal_coverage", value=0.5),
            ),
        },
        {
            "metric_id": "sciplex3.marginal-interval-width-p80",
            "family": MetricFamily.CALIBRATION,
            "formula": (
                "mean_well_feature(panel_logcp10k_central_80_percent_predictive_interval_width)"
            ),
            "units": "panel_logCP10k_interval_width",
            "weighting": MetricWeightingScheme.EQUAL_EVALUATION_UNIT,
            "parameters": (
                BenchmarkParameter(name="calibration_fit_partition", value="calibration"),
                BenchmarkParameter(name="diagnostic", value=True),
                BenchmarkParameter(name="nominal_coverage", value=0.8),
            ),
        },
        {
            "metric_id": "sciplex3.marginal-interval-width-p95",
            "family": MetricFamily.CALIBRATION,
            "formula": (
                "mean_well_feature(panel_logcp10k_central_95_percent_predictive_interval_width)"
            ),
            "units": "panel_logCP10k_interval_width",
            "weighting": MetricWeightingScheme.EQUAL_EVALUATION_UNIT,
            "parameters": (
                BenchmarkParameter(name="calibration_fit_partition", value="calibration"),
                BenchmarkParameter(name="diagnostic", value=True),
                BenchmarkParameter(name="nominal_coverage", value=0.95),
            ),
        },
        {
            "metric_id": "sciplex3.marginal-crps-logcp10k",
            "family": MetricFamily.PREDICTIVE_PROPER_SCORE,
            "formula": (
                "mean_well_feature(E_abs(X-Y)-0.5*E_abs(X-X_prime))_after_log1p_"
                "10000_count_over_exact_declared_2000_panel_total"
            ),
            "units": "panel_logCP10k",
            "weighting": MetricWeightingScheme.EQUAL_EVALUATION_UNIT,
            "parameters": (
                BenchmarkParameter(name="feature_count", value=2000),
                BenchmarkParameter(name="feature_panel_fit_partition", value="train"),
            ),
        },
        {
            "metric_id": "sciplex3.vehicle-relative-pseudobulk-rmse",
            "family": MetricFamily.INTERVENTION_EFFECT,
            "formula": (
                "mean_treated_well(rmse_feature(predicted_panel_logcp10k_effect,"
                "observed_panel_logcp10k_well_minus_same_plate_vehicle_effect))"
            ),
            "units": "panel_logCP10k_effect_rmse",
            "weighting": MetricWeightingScheme.EQUAL_EVALUATION_UNIT,
            "parameters": (
                BenchmarkParameter(name="control_matching_stratum", value="plate"),
                BenchmarkParameter(name="feature_panel_fit_partition", value="train"),
            ),
        },
    )
    definitions = []
    for spec in specs:
        metric_id = spec["metric_id"]
        binding = binding_by_metric_id[metric_id]
        weighting_scheme = spec["weighting"]
        definitions.append(
            BenchmarkMetricDefinition(
                metric_id=metric_id,
                metric_version="1.1.0-definition",
                family=spec["family"],
                query_fingerprint=query.fingerprint,
                evidence_binding_ids=(binding.binding_id,),
                evaluation_partition_ids=(test_partition_id,),
                target_output_keys=(TARGET_TERM.key,),
                horizon_names=(HORIZON_NAME,),
                implementation_binding=_specification_only(
                    support["metric_suite"],
                    component=f"{metric_id} metric",
                ),
                prediction_representation=PredictionRepresentation.POSTERIOR_SAMPLES,
                target_representation=TargetRepresentation.COUNT_MATRIX,
                evaluation_unit=_unit(ExperimentalUnitLevel.WELL),
                aggregation=MetricAggregation.MEAN,
                weighting=MetricWeightingPolicy(
                    scheme=weighting_scheme,
                    group_dependence_id=(
                        "compound"
                        if weighting_scheme
                        is MetricWeightingScheme.EQUAL_GROUP_THEN_EQUAL_EVALUATION_UNIT
                        else None
                    ),
                ),
                direction=MetricDirection.MINIMIZE,
                units=spec["units"],
                formula=spec["formula"],
                parameters=tuple(
                    sorted(
                        (
                            *spec["parameters"],
                            BenchmarkParameter(
                                name="scoring_denominator_scope",
                                value="declared_ordered_2000_feature_panel_only",
                            ),
                            BenchmarkParameter(
                                name="scoring_transform_sha256",
                                value=support["scoring_transform"].sha256,
                            ),
                            BenchmarkParameter(
                                name="zero_panel_total_policy",
                                value="error_fail_evaluation_no_exclusion_or_imputation",
                            ),
                        ),
                        key=lambda parameter: parameter.name,
                    )
                ),
                missingness_policy=MetricMissingnessPolicy.ERROR_ON_MISSING,
                uncertainty=_metric_uncertainty(metric_id, support),
                minimum_evaluation_units=192,
            )
        )
    return tuple(sorted(definitions, key=lambda item: item.metric_id))


def build_baselines(
    query: StateQuery,
    split_plan: BenchmarkSplitPlan,
    support: dict[str, ContentAddressedArtifact],
) -> tuple[BenchmarkBaselineDefinition, ...]:
    train_partition = next(
        item.partition_id
        for item in split_plan.partitions
        if item.role is BenchmarkPartitionRole.TRAIN
    )
    specs = (
        ("exact-condition-negative-binomial", False, 1),
        ("exact-condition-rep1-empirical-resampling", False, 1),
        ("hierarchical-well-negative-binomial", False, 1),
        ("low-rank-compound-dose-response", False, 1),
        ("matched-vehicle-resampling", False, 1),
        ("nearest-supported-dose", False, 1),
        ("persistence", True, 1),
        ("temporal-state-space", True, 2),
    )
    baselines = []
    for baseline_id, needs_pre_cutoff_target, horizon_count in specs:
        baselines.append(
            BenchmarkBaselineDefinition(
                baseline_id=baseline_id,
                baseline_version="1.0.0-definition",
                query_fingerprint=query.fingerprint,
                implementation_binding=_specification_only(
                    support["baseline_suite"],
                    component=f"{baseline_id} baseline",
                ),
                applicability=BaselineApplicabilityRule(
                    allowed_subject_kinds=(SubjectKind.POPULATION,),
                    requires_intervention_space=True,
                    requires_pre_cutoff_target_observation=needs_pre_cutoff_target,
                    minimum_target_count=1,
                    minimum_horizon_count=horizon_count,
                ),
                training_partition_ids=(train_partition,),
                parameters=(BenchmarkParameter(name="well_is_replication_unit", value=True),),
                seeds=(0, 1, 2, 3, 4),
            )
        )
    return tuple(sorted(baselines, key=lambda item: item.baseline_id))


def _metric_reference(metric: BenchmarkMetricDefinition) -> MetricDefinitionReference:
    return MetricDefinitionReference(
        metric_id=metric.metric_id,
        metric_version=metric.metric_version,
        metric_fingerprint=metric.fingerprint,
    )


def _baseline_reference(
    baseline: BenchmarkBaselineDefinition,
) -> BaselineDefinitionReference:
    return BaselineDefinitionReference(
        baseline_id=baseline.baseline_id,
        baseline_version=baseline.baseline_version,
        baseline_fingerprint=baseline.fingerprint,
    )


def build_acceptance(
    metrics: tuple[BenchmarkMetricDefinition, ...],
    baselines: tuple[BenchmarkBaselineDefinition, ...],
    split_plan: BenchmarkSplitPlan,
) -> tuple[tuple[BenchmarkAcceptanceRule, ...], BenchmarkAcceptancePolicy]:
    metric_by_id = {metric.metric_id: metric for metric in metrics}
    baseline_by_id = {baseline.baseline_id: baseline for baseline in baselines}
    test_partition_id = next(
        partition.partition_id
        for partition in split_plan.partitions
        if partition.role is BenchmarkPartitionRole.UNTOUCHED_TEST
    )
    mandatory_baselines = tuple(
        baseline_by_id[baseline_id] for baseline_id in MANDATORY_PROBABILISTIC_BASELINE_IDS
    )
    best_mandatory = BestApplicableBaselineComparator(
        baselines=tuple(_baseline_reference(baseline) for baseline in mandatory_baselines)
    )
    rules: list[BenchmarkAcceptanceRule] = []
    crps = metric_by_id["sciplex3.marginal-crps-logcp10k"]
    for baseline in mandatory_baselines:
        rules.append(
            BenchmarkAcceptanceRule(
                rule_id=f"crps-superior--{baseline.baseline_id}",
                metric=_metric_reference(crps),
                partition_id=test_partition_id,
                comparison=ThresholdComparison.LESS_THAN,
                estimate=ThresholdEstimate.UPPER_CONFIDENCE_BOUND,
                baseline_margin=0.0,
                baseline_comparator=ExactBaselineComparator(baseline=_baseline_reference(baseline)),
                baseline_margin_mode=BaselineMarginMode.ABSOLUTE_DIFFERENCE,
                baseline_requirement=BaselineRequirement.SUPERIOR,
                confidence_level=0.95,
                rationale=(
                    "The upper one-sided 95% confidence bound for the paired compound-block "
                    "candidate-minus-baseline marginal CRPS effect must be strictly below zero."
                ),
            )
        )

    joint_effect_metric_ids = (
        "sciplex3.joint-energy-train-pca",
        "sciplex3.vehicle-relative-pseudobulk-rmse",
    )
    for metric_id in joint_effect_metric_ids:
        metric = metric_by_id[metric_id]
        rules.extend(
            (
                BenchmarkAcceptanceRule(
                    rule_id=f"best-baseline-noninferior--{metric_id}",
                    metric=_metric_reference(metric),
                    partition_id=test_partition_id,
                    comparison=ThresholdComparison.LESS_THAN_OR_EQUAL,
                    estimate=ThresholdEstimate.UPPER_CONFIDENCE_BOUND,
                    baseline_margin=0.02,
                    baseline_comparator=best_mandatory,
                    baseline_margin_mode=BaselineMarginMode.RELATIVE_FRACTION,
                    baseline_requirement=BaselineRequirement.NONINFERIOR,
                    confidence_level=0.95,
                    rationale=(
                        "The upper one-sided 95% confidence bound for the paired compound-block "
                        "candidate-minus-best-baseline relative effect must be no greater than "
                        "0.02."
                    ),
                ),
                BenchmarkAcceptanceRule(
                    rule_id=f"best-baseline-superior--{metric_id}",
                    metric=_metric_reference(metric),
                    partition_id=test_partition_id,
                    comparison=ThresholdComparison.LESS_THAN,
                    estimate=ThresholdEstimate.UPPER_CONFIDENCE_BOUND,
                    baseline_margin=0.0,
                    baseline_comparator=best_mandatory,
                    baseline_margin_mode=BaselineMarginMode.RELATIVE_FRACTION,
                    baseline_requirement=BaselineRequirement.SUPERIOR,
                    confidence_level=0.95,
                    rationale=(
                        "At least one paired compound-block candidate-minus-best-baseline relative "
                        "effect for joint energy or vehicle-relative effect RMSE must have an "
                        "upper one-sided 95% confidence bound strictly below zero."
                    ),
                ),
            )
        )

    coverage_rule_ids = []
    for probability in (50, 80, 95):
        metric = metric_by_id[f"sciplex3.marginal-coverage-error-p{probability}"]
        rule_id = f"absolute-coverage-error-p{probability}"
        coverage_rule_ids.append(rule_id)
        rules.append(
            BenchmarkAcceptanceRule(
                rule_id=rule_id,
                metric=_metric_reference(metric),
                partition_id=test_partition_id,
                comparison=ThresholdComparison.LESS_THAN_OR_EQUAL,
                estimate=ThresholdEstimate.UPPER_CONFIDENCE_BOUND,
                absolute_threshold=0.03,
                confidence_level=0.95,
                rationale=(
                    f"The upper 95% confidence bound on absolute marginal coverage error at "
                    f"nominal {probability / 100:.2f} must not exceed 0.03."
                ),
            )
        )

    rules_tuple = tuple(sorted(rules, key=lambda item: item.rule_id))
    crps_rule_ids = tuple(
        sorted(rule.rule_id for rule in rules if rule.rule_id.startswith("crps-superior--"))
    )
    noninferiority_rule_ids = tuple(
        sorted(
            rule.rule_id for rule in rules if rule.rule_id.startswith("best-baseline-noninferior--")
        )
    )
    superiority_rule_ids = tuple(
        sorted(
            rule.rule_id for rule in rules if rule.rule_id.startswith("best-baseline-superior--")
        )
    )
    policy = BenchmarkAcceptancePolicy(
        policy_id="sciplex3-k562-component-acceptance",
        policy_version="1.0.0",
        root_group_id="all-primary-gates",
        groups=tuple(
            sorted(
                (
                    BenchmarkAcceptanceGroup(
                        group_id="all-primary-gates",
                        operator=AcceptanceGroupOperator.ALL,
                        child_group_ids=(
                            "coverage-error-all",
                            "crps-mandatory-all",
                            "joint-effect-noninferior-all",
                            "joint-effect-superior-any",
                        ),
                    ),
                    BenchmarkAcceptanceGroup(
                        group_id="coverage-error-all",
                        operator=AcceptanceGroupOperator.ALL,
                        rule_ids=tuple(sorted(coverage_rule_ids)),
                    ),
                    BenchmarkAcceptanceGroup(
                        group_id="crps-mandatory-all",
                        operator=AcceptanceGroupOperator.ALL,
                        rule_ids=crps_rule_ids,
                    ),
                    BenchmarkAcceptanceGroup(
                        group_id="joint-effect-noninferior-all",
                        operator=AcceptanceGroupOperator.ALL,
                        rule_ids=noninferiority_rule_ids,
                    ),
                    BenchmarkAcceptanceGroup(
                        group_id="joint-effect-superior-any",
                        operator=AcceptanceGroupOperator.ANY,
                        rule_ids=superiority_rule_ids,
                    ),
                ),
                key=lambda item: item.group_id,
            )
        ),
    )
    return rules_tuple, policy


def build_evaluation_case_set(
    query: StateQuery,
    split_plan: BenchmarkSplitPlan,
    prep: dict[str, Any],
) -> BenchmarkEvaluationCaseSet:
    spec_by_id = {spec.spec_id: spec for spec in query.intervention_space}
    role_partition_ids = {
        partition.role.value: partition.partition_id for partition in split_plan.partitions
    }
    role_key_by_partition_role = {
        BenchmarkPartitionRole.TRAIN.value: "train",
        BenchmarkPartitionRole.CALIBRATION.value: "calibration",
        BenchmarkPartitionRole.MODEL_SELECTION_VALIDATION.value: ("model_selection_validation"),
        BenchmarkPartitionRole.UNTOUCHED_TEST.value: "untouched_test",
    }
    controls_by_plate = {
        plate: tuple(
            sorted(
                group["composite_well_id"]
                for group in prep["groups"]["groups"]
                if group["plate"] == plate and group["is_vehicle_control"]
            )
        )
        for plate in sorted({group["plate"] for group in prep["groups"]["groups"]})
    }
    contexts = []
    context_by_plate: dict[str, EvaluationContextBinding] = {}
    stratum_fingerprint_by_plate: dict[str, str] = {}
    for plate in sorted(controls_by_plate):
        path = SUPPORT_DIR / "contexts" / f"{plate}.json"
        context_artifact = _artifact_for_file(f"sciplex3-static-context-{plate}", path)
        context = EvaluationContextBinding(
            context_id=f"static-context--{plate}",
            context_fingerprint=context_artifact.sha256,
            context_artifact=context_artifact,
        )
        contexts.append(context)
        context_by_plate[plate] = context
        stratum_fingerprint_by_plate[plate] = canonical_fingerprint(
            {
                "identity_expression_fingerprint": PLATE_IDENTITY.fingerprint,
                "level": ExperimentalUnitLevel.PLATE.value,
                "value": plate,
            }
        )
    cases = []
    multiplicities: dict[str, int] = {}
    for group in prep["groups"]["groups"]:
        well_id = group["composite_well_id"]
        role = group["partition_role"]
        partition_id = role_partition_ids[role]
        context = context_by_plate[group["plate"]]
        if group["is_vehicle_control"]:
            intervention_ids: tuple[str, ...] = ()
            case_role = EvaluationCaseRole.MATCHED_NO_ACTION_CONTROL
            matched_controls: tuple[str, ...] = ()
        else:
            expected_spec_id = (
                f"{_slug(group['normalized_perturbation_label'])}-"
                f"{int(group['dose_value_nm']):05d}-nm"
            )
            spec = spec_by_id[expected_spec_id]
            intervention_ids = (spec.spec_id,)
            multiplicities[spec.spec_id] = multiplicities.get(spec.spec_id, 0) + 1
            case_role = EvaluationCaseRole.TREATED
            matched_controls = controls_by_plate[group["plate"]]
        cases.append(
            BenchmarkEvaluationCase(
                case_id=f"well-case--{hashlib.sha256(well_id.encode('utf-8')).hexdigest()}",
                partition_id=partition_id,
                evaluation_unit_id=well_id,
                prediction_subject_id=f"sciplex3-k562-t0-assigned-well-population--{well_id}",
                context_id=context.context_id,
                context_fingerprint=context.context_fingerprint,
                matching_stratum_id=group["plate"],
                matching_stratum_fingerprint=stratum_fingerprint_by_plate[group["plate"]],
                intervention_spec_ids=intervention_ids,
                horizon_name=HORIZON_NAME,
                target_output_keys=(TARGET_TERM.key,),
                role=case_role,
                matched_control_evaluation_unit_ids=matched_controls,
            )
        )
    cases.sort(key=lambda item: item.case_id)
    case_path = SUPPORT_DIR / "evaluation-cases.json"
    _write_canonical(
        case_path,
        [case.model_dump(mode="json") for case in cases],
    )
    partition_memberships = []
    for partition_role, partition_id in sorted(role_partition_ids.items()):
        role = role_key_by_partition_role[partition_role]
        partition = prep["partitions_by_role"][role]
        partition_memberships.append(
            EvaluationCasePartitionBinding(
                partition_id=partition_id,
                evaluation_unit_ids=_id_membership(
                    role,
                    "well-ids",
                    partition["well_count"],
                ),
            )
        )
    return BenchmarkEvaluationCaseSet(
        case_set_id="sciplex3-k562-exact-well-cases",
        case_set_version="1.0.0",
        query_fingerprint=query.fingerprint,
        evaluation_unit=_unit(ExperimentalUnitLevel.WELL),
        contexts=tuple(sorted(contexts, key=lambda item: item.context_id)),
        case_artifact=_artifact_for_file("sciplex3-k562-exact-well-evaluation-cases", case_path),
        case_count=len(cases),
        partition_memberships=tuple(
            sorted(partition_memberships, key=lambda item: item.partition_id)
        ),
        intervention_case_counts=tuple(
            EvaluationInterventionMultiplicity(
                intervention_spec_id=spec_id,
                case_count=count,
            )
            for spec_id, count in sorted(multiplicities.items())
        ),
        no_action_control_case_count=sum(
            case.role is EvaluationCaseRole.MATCHED_NO_ACTION_CONTROL for case in cases
        ),
        cases=tuple(cases),
    )


def build_definition(
    query: StateQuery,
    query_artifact: ContentAddressedArtifact,
    evidence_bindings: tuple[BenchmarkEvidenceBinding, ...],
    split_plan: BenchmarkSplitPlan,
    case_set: BenchmarkEvaluationCaseSet,
    metrics: tuple[BenchmarkMetricDefinition, ...],
    baselines: tuple[BenchmarkBaselineDefinition, ...],
    acceptance_rules: tuple[BenchmarkAcceptanceRule, ...],
    acceptance_policy: BenchmarkAcceptancePolicy,
) -> BenchmarkDefinition:
    query_binding = StateQueryBinding(
        query_id=QUERY_ID,
        query_version="1.0.0",
        query_schema_version="2.0",
        query_fingerprint=query.fingerprint,
        query_artifact=query_artifact,
        state_query=query,
    )
    scope = BenchmarkScope(
        scope_id="sciplex3-k562-context-to-24h-endpoint",
        query_fingerprint=query.fingerprint,
        subject_kind=query.subject.kind,
        system_boundary=query.system_boundary,
        biological_system=query.subject.biological_system,
        target_output_keys=tuple(sorted(output.term.key for output in query.target_outputs)),
        horizon_names=tuple(sorted(horizon.name for horizon in query.prediction_horizons)),
        intervention_spec_ids=tuple(sorted(spec.spec_id for spec in query.intervention_space)),
        scientific_claims=(ScientificClaim.INTERVENTION_EFFECT,),
        inference_cutoff_seconds=0.0,
        reference_estimand_causal_status=CausalStatus.IDENTIFIED_POPULATION_EFFECT,
        forecast_causal_status=CausalStatus.PREDICTIVE_ASSOCIATION,
        estimand=(
            "within-study assignment intention-to-treat contrast for each exact compound-dose "
            "well versus same-plate no-action vehicle wells on the recovered-nucleus 24-hour "
            "count-vector population distribution"
        ),
    )
    return BenchmarkDefinition(
        benchmark_id=BENCHMARK_ID,
        benchmark_version="1.0.0",
        title="sci-Plex3 K562 context-to-24-hour replicate-transfer component benchmark",
        description=(
            "A frozen, public-real-data component benchmark for query-conditioned prediction of "
            "the destructive 24-hour recovered-nucleus RNA count distribution on an ordered "
            "train-derived 2,000-feature panel after one exact source-scoped small-molecule "
            "exposure."
        ),
        design_status=BenchmarkLifecycle.FROZEN,
        intent=BenchmarkIntent.COMPONENT_BENCHMARK,
        query=query_binding,
        scope=scope,
        evidence_bindings=evidence_bindings,
        split_plan=split_plan,
        evaluation_case_set=case_set,
        metrics=metrics,
        baselines=baselines,
        acceptance_rules=acceptance_rules,
        acceptance_policy=acceptance_policy,
        notes=tuple(
            sorted(
                (
                    "All model, metric, uncertainty, and applicable baseline executions remain "
                    "blocked until executable implementations and golden fixtures exist.",
                    "Compound labels are source-scoped administered agents only; no chemical "
                    "ontology identity, biological target, or target-engagement claim is made.",
                    "The benchmark has no pre-cutoff molecular observation, tracked cell, "
                    "viability endpoint, second horizon, or external-study transport evidence.",
                    "The reference assignment contrast is identified within this study; a model "
                    "forecast remains a predictive association.",
                    "The state subject is the K562 population assigned to a source well at t=0; "
                    "only the future target is conditioned on nuclei recovered at 24 hours.",
                    "Scoring is computable from the declared 2,000 raw-count coordinates alone: "
                    "the denominator is their per-sample sum; no full source-axis count is read, "
                    "and an invalid or zero-total sample fails evaluation without exclusion or "
                    "imputation.",
                    "Vehicle background is static context shared by treated and control wells; "
                    "matched no-action controls mean zero modeled active compound, not literal "
                    "absence of vehicle.",
                )
            )
        ),
    )


def build_leakage_audit(
    split_plan: BenchmarkSplitPlan,
    support: dict[str, ContentAddressedArtifact],
) -> BenchmarkLeakageAudit:
    all_partition_ids = tuple(item.partition_id for item in split_plan.partitions)
    report_locator = support["leakage_evidence"].uri
    checks: list[LeakageAuditCheck] = []
    for closure in split_plan.protected_group_closures:
        partition_ids = tuple(
            item.partition_id
            for item in split_plan.partitions
            if item.physical_dataset_binding_id == closure.physical_dataset_binding_id
        )
        checks.append(
            LeakageAuditCheck(
                check_id=f"record-disjoint--{closure.physical_dataset_binding_id}",
                kind=LeakageCheckKind.RECORD_MEMBERSHIP_DISJOINT,
                status=AuditCheckStatus.PASSED,
                physical_dataset_binding_id=closure.physical_dataset_binding_id,
                partition_ids=partition_ids,
                comparisons_expected=6,
                comparisons_completed=6,
                overlap_count=0,
                report_locator=report_locator,
                notes=("All six physical partition pairs have zero exact source-record overlap.",),
            )
        )
        for group in closure.protected_groups:
            checks.append(
                LeakageAuditCheck(
                    check_id=(
                        f"protected-{group.unit.level.value}-disjoint--"
                        f"{closure.physical_dataset_binding_id}"
                    ),
                    kind=LeakageCheckKind.PROTECTED_GROUP_DISJOINT,
                    status=AuditCheckStatus.PASSED,
                    physical_dataset_binding_id=closure.physical_dataset_binding_id,
                    protected_unit=group.unit,
                    partition_ids=partition_ids,
                    comparisons_expected=6,
                    comparisons_completed=6,
                    overlap_count=0,
                    report_locator=report_locator,
                    notes=(
                        "All six physical partition pairs have zero "
                        f"{group.unit.level.value} identity overlap.",
                    ),
                )
            )
    all_pairs = len(all_partition_ids) * (len(all_partition_ids) - 1) // 2
    checks.extend(
        (
            LeakageAuditCheck(
                check_id="global-preprocessing-fit-isolated",
                kind=LeakageCheckKind.PREPROCESSING_FIT_ISOLATED,
                status=AuditCheckStatus.PASSED,
                partition_ids=all_partition_ids,
                report_locator=report_locator,
                notes=(
                    "Feature selection and its full-axis transform accessed 94,785 train rows and "
                    "zero held-out count rows. The fixed scoring transform has no fitted state and "
                    "uses only the declared 2,000 coordinates of each evaluated sample.",
                ),
            ),
            LeakageAuditCheck(
                check_id="global-source-duplicate-review",
                kind=LeakageCheckKind.SOURCE_DUPLICATE_DISJOINT,
                status=AuditCheckStatus.NOT_ASSESSED,
                partition_ids=all_partition_ids,
                comparisons_expected=all_pairs,
                comparisons_completed=all_pairs,
                overlap_count=0,
                report_locator=report_locator,
                blockers=(
                    "Exact source-record identity is disjoint in all six physical partition "
                    "pairs, but biological duplicate detection beyond exact source IDs has not "
                    "run.",
                ),
            ),
            LeakageAuditCheck(
                check_id="global-target-derivation-isolated",
                kind=LeakageCheckKind.TARGET_DERIVATION_ISOLATED,
                status=AuditCheckStatus.PASSED,
                partition_ids=all_partition_ids,
                report_locator=report_locator,
                notes=(
                    "The benchmark target is the observed raw integer endpoint count vector; no "
                    "fitted target-label transformation uses held-out outcomes.",
                ),
            ),
            LeakageAuditCheck(
                check_id="global-temporal-cutoff-respected",
                kind=LeakageCheckKind.TEMPORAL_CUTOFF_RESPECTED,
                status=AuditCheckStatus.PASSED,
                partition_ids=all_partition_ids,
                report_locator=report_locator,
                notes=(
                    "The query admits zero required molecular observations at t=0, excludes the "
                    "RNA target modality from inference evidence, and locates RNA only at "
                    "t=86,400 seconds.",
                ),
            ),
        )
    )
    preparation_code = _artifact_for_file(
        "sciplex3-preparation-generator",
        REPO_ROOT / "scripts/prepare_sciplex3_k562.py",
        media_type="text/x-python",
    )
    return BenchmarkLeakageAudit(
        audit_id="sciplex3-k562-definition-leakage-audit",
        audit_version="1.0.0",
        split_plan_fingerprint=split_plan.fingerprint,
        implementation=VersionedImplementation(
            implementation_id="cellstate.prepare-sciplex3-k562",
            implementation_version="1.0.0",
            code_artifact=preparation_code,
            entrypoint="scripts.prepare_sciplex3_k562:main",
            runtime="python-3.11+h5py-3.x+numpy-2.x",
        ),
        report_artifact=support["leakage_evidence"],
        evaluated_partition_ids=all_partition_ids,
        checks=tuple(sorted(checks, key=lambda item: item.check_id)),
        reviewed_by=("cellstate-maintainers",),
        reviewed_on=date(2026, 8, 9),
    )


def _resolution_fingerprint(value: Any) -> str:
    return canonical_fingerprint(value.model_dump(mode="json"))


def build_admission(
    definition: BenchmarkDefinition,
    manifest: DatasetManifest,
    leakage_audit: BenchmarkLeakageAudit,
) -> BenchmarkAdmission:
    evidence_resolutions = []
    for binding in definition.evidence_bindings:
        resolution = manifest.resolve_assessment(
            binding.assessment_reference,
            use_case=DataUseCase.BENCHMARK_EVALUATION,
        )
        evidence_resolutions.append(
            EvidenceResolutionBinding(
                evidence_binding_id=binding.binding_id,
                assessment_reference=binding.assessment_reference,
                resolution_fingerprint=_resolution_fingerprint(resolution),
            )
        )
    runs = []
    for baseline in definition.baselines:
        reference = _baseline_reference(baseline)
        if baseline.applicability.applies_to(definition.query.state_query):
            runs.append(
                BaselineRun(
                    baseline=reference,
                    status=BaselineRunStatus.NOT_RUN,
                    applicability_rule_fingerprint=baseline.applicability.fingerprint,
                    blockers=(
                        "The applicable baseline has a frozen definition but has not passed "
                        "executable golden cases or produced predictions and well-clustered "
                        "metric results.",
                    ),
                )
            )
        else:
            explanation = (
                "The frozen query has no required pre-cutoff target-modality observation."
                if baseline.baseline_id == "persistence"
                else (
                    "The frozen query has no required pre-cutoff target-modality observation and "
                    "only one future horizon."
                )
            )
            runs.append(
                BaselineRun(
                    baseline=reference,
                    status=BaselineRunStatus.NOT_APPLICABLE,
                    applicability_rule_fingerprint=baseline.applicability.fingerprint,
                    notes=(explanation,),
                )
            )
    return BenchmarkAdmission(
        admission_id="sciplex3-k562-component-freeze",
        admission_version="1.0.0",
        definition_fingerprint=definition.fingerprint,
        status=BenchmarkAdmissionStatus.COMPONENT_BENCHMARK,
        evidence_resolutions=tuple(
            sorted(evidence_resolutions, key=lambda item: item.evidence_binding_id)
        ),
        leakage_audit_fingerprint=leakage_audit.fingerprint,
        baseline_runs=tuple(sorted(runs, key=lambda item: item.baseline.baseline_id)),
        paired_baseline_comparisons=(),
        reasons=tuple(
            sorted(
                (
                    "Applicable mandatory baselines and benchmark metric implementations have not "
                    "been executed against golden cases or protected evaluation partitions.",
                    "This endpoint-only component has no pre-intervention molecular measurement, "
                    "same-cell linkage, viability target, second horizon, or external-study "
                    "transport evidence.",
                    "The benchmark freezes context-to-24-hour recovered-nucleus assay-response "
                    "semantics only and cannot admit the full cell-state system or intervention "
                    "planning.",
                    "The source-duplicate audit remains explicitly unassessed beyond exact source "
                    "record identities.",
                )
            )
        ),
        reviewed_by=("cellstate-maintainers",),
        reviewed_on=date(2026, 8, 9),
    )


def main() -> None:
    prep = _load_preparation()
    support = _write_support_specs(prep)
    manifest = build_manifest(prep)
    query = build_query(prep["labels"], support)

    _write_canonical(MANIFEST_PATH, manifest)
    _write_canonical(QUERY_PATH, query)
    if MANIFEST_PATH.read_bytes() != manifest.canonical_json_bytes:
        raise RuntimeError("reviewed manifest is not the exact canonical manifest payload")
    if _sha256_bytes(QUERY_PATH.read_bytes()) != query.fingerprint:
        raise RuntimeError("StateQuery artifact is not the exact canonical query payload")

    parsed_manifest = DatasetManifest.model_validate_json(MANIFEST_PATH.read_bytes())
    parsed_query = StateQuery.model_validate_json(QUERY_PATH.read_bytes())
    if parsed_manifest.fingerprint != manifest.fingerprint or parsed_query.fingerprint != (
        query.fingerprint
    ):
        raise RuntimeError("serialized manifest or query changed identity on strict reload")

    manifest_artifact = _artifact_for_file(
        "sciplex3-k562-reviewed-canonical-manifest",
        MANIFEST_PATH,
    )
    query_artifact = _artifact_for_file(
        "sciplex3-k562-frozen-state-query-v2",
        QUERY_PATH,
    )
    evidence_bindings = build_evidence_bindings(
        manifest,
        query,
        manifest_artifact,
        support,
    )
    split_plan = build_split_plan(manifest, query, evidence_bindings, prep)
    metrics = build_metrics(query, evidence_bindings, split_plan, support)
    baselines = build_baselines(query, split_plan, support)
    acceptance_rules, acceptance_policy = build_acceptance(
        metrics,
        baselines,
        split_plan,
    )
    case_set = build_evaluation_case_set(query, split_plan, prep)
    definition = build_definition(
        query,
        query_artifact,
        evidence_bindings,
        split_plan,
        case_set,
        metrics,
        baselines,
        acceptance_rules,
        acceptance_policy,
    )
    leakage_audit = build_leakage_audit(split_plan, support)
    admission = build_admission(definition, manifest, leakage_audit)
    artifact = BenchmarkArtifact(
        definition=definition,
        leakage_audit=leakage_audit,
        admission=admission,
    )
    _write_canonical(BENCHMARK_PATH, artifact)
    parsed_artifact = BenchmarkArtifact.model_validate_json(BENCHMARK_PATH.read_bytes())
    if parsed_artifact.fingerprint != artifact.fingerprint:
        raise RuntimeError("serialized benchmark changed identity on strict reload")

    verification = verify_benchmark_artifact(
        parsed_artifact,
        {binding.binding_id: parsed_manifest for binding in evidence_bindings},
    )
    if not verification.assessment_and_permission_gates_passed:
        raise RuntimeError("frozen assessment or permission gates did not re-resolve")
    if verification.performance_gates_passed or verification.admission_ready:
        raise RuntimeError("unexecuted component benchmark cannot pass performance admission")
    if (
        not verification.verified
        or verification.declared_status is not BenchmarkAdmissionStatus.COMPONENT_BENCHMARK
    ):
        raise RuntimeError("frozen component benchmark did not verify at its declared boundary")

    print(
        json.dumps(
            {
                "admission_ready": verification.admission_ready,
                "artifact_fingerprint": artifact.fingerprint,
                "assessment_and_permission_gates_passed": (
                    verification.assessment_and_permission_gates_passed
                ),
                "benchmark_path": str(BENCHMARK_PATH.relative_to(REPO_ROOT)),
                "case_count": case_set.case_count,
                "declared_status": admission.status.value,
                "manifest_fingerprint": manifest.fingerprint,
                "metric_count": len(metrics),
                "performance_gates_passed": verification.performance_gates_passed,
                "query_fingerprint": query.fingerprint,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
