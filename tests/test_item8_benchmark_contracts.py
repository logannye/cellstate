from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import TypeVar

import pytest
from pydantic import BaseModel, ValidationError
from test_item6_eligibility import manifest_factory

import cellstate.data as data
from cellstate.domain import CausalStatus, OntologyTerm, StateQuery, SystemBoundary
from cellstate.domain.common import canonical_fingerprint, canonical_json_bytes
from cellstate.domain.query import PredictionHorizon, Timescale
from cellstate.domain.subjects import (
    AggregationStatistic,
    IdentityBasis,
    SubjectKind,
    SubjectSpecification,
    TargetAggregation,
)
from cellstate.training import LossKind

ModelT = TypeVar("ModelT", bound=BaseModel)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def artifact(
    artifact_id: str,
    *,
    sha256: str | None = None,
    byte_count: int = 128,
    media_type: str = "application/json",
) -> data.ContentAddressedArtifact:
    return data.ContentAddressedArtifact(
        artifact_id=artifact_id,
        uri=f"https://example.org/frozen/{artifact_id}",
        sha256=sha256 or digest(artifact_id),
        byte_count=byte_count,
        media_type=media_type,
    )


def implementation(implementation_id: str) -> data.VersionedImplementation:
    return data.VersionedImplementation(
        implementation_id=implementation_id,
        implementation_version="1.0.0",
        code_artifact=artifact(f"{implementation_id}-code", media_type="application/zip"),
        entrypoint=f"{implementation_id}:run",
        runtime="python-3.11",
    )


def executable_binding(name: str) -> data.ExecutableImplementationBinding:
    return data.ExecutableImplementationBinding(
        specification_artifact=artifact(f"{name}-specification"),
        implementation=implementation(name),
        golden_fixture_artifact=artifact(f"{name}-golden"),
    )


def planned_binding(name: str) -> data.SpecificationOnlyImplementationBinding:
    return data.SpecificationOnlyImplementationBinding(
        specification_artifact=artifact(f"{name}-specification"),
        blockers=("Executable implementation and golden fixture are not yet frozen.",),
    )


def query_binding(query: StateQuery) -> data.StateQueryBinding:
    query_bytes = json.dumps(
        query.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return data.StateQueryBinding(
        query_id="synthetic-state-query",
        query_version="1.0.0",
        query_schema_version=query.schema_version,
        query_fingerprint=query.fingerprint,
        query_artifact=artifact(
            "synthetic-state-query-canonical",
            sha256=query.fingerprint,
            byte_count=len(query_bytes),
        ),
        state_query=query,
    )


def assessment_reference(
    assessment_id: str,
    manifest_fingerprint: str,
) -> data.DatasetAssessmentReference:
    return data.DatasetAssessmentReference(
        dataset_manifest_fingerprint=manifest_fingerprint,
        assessment_id=assessment_id,
        assessment_fingerprint=digest(assessment_id),
    )


def evidence_binding(
    binding_id: str,
    *,
    query: StateQuery,
    kind: data.AssessmentKind,
    manifest_fingerprint: str,
) -> data.BenchmarkEvidenceBinding:
    target_mappings = tuple(
        data.EvidenceTargetMapping(
            target_output_key=output.term.key,
            target_output_fingerprint=canonical_fingerprint(output.model_dump(mode="json")),
            target_units=output.units,
            target_aggregation=output.aggregation,
            aggregation_unit=unit(data.ExperimentalUnitLevel.WELL, "well_id"),
            assessment_modalities=(output.term.key,),
            semantics_artifact=artifact(f"{binding_id}-{output.term.key}-semantics"),
        )
        for output in sorted(query.target_outputs, key=lambda item: item.term.key)
    )
    intervention_mappings = tuple(
        data.EvidenceInterventionMapping(
            intervention_spec_id=spec.spec_id,
            intervention_spec_fingerprint=canonical_fingerprint(spec.model_dump(mode="json")),
            assessment_intervention_kind_key=spec.kind.key,
            domain_mapping_artifact=artifact(f"{binding_id}-{spec.spec_id}-action-domain"),
        )
        for spec in sorted(query.intervention_space, key=lambda item: item.spec_id)
    )
    environment_mappings = tuple(
        data.EvidenceEnvironmentMapping(
            environment_variable_key=spec.variable.key,
            environment_spec_fingerprint=canonical_fingerprint(spec.model_dump(mode="json")),
            assessment_environment_variable_key=spec.variable.key,
            domain_mapping_artifact=artifact(
                f"{binding_id}-{spec.variable.key}-environment-domain"
            ),
        )
        for spec in sorted(query.environment_space, key=lambda item: item.variable.key)
    )
    if kind is data.AssessmentKind.LOSS:
        assessment_identity: data.AssessmentIdentity = data.LossAssessmentIdentity(
            loss_kind=LossKind.MULTI_HORIZON_FUTURE,
        )
    elif kind is data.AssessmentKind.METRIC:
        assessment_identity = data.MetricAssessmentIdentity(
            metric_id="well-weighted-proper-score",
            metric_family=data.MetricFamily.PREDICTIVE_PROPER_SCORE,
            partition_purpose=data.MetricPartitionPurpose.UNTOUCHED_TEST,
        )
    else:
        assessment_identity = data.ClaimAssessmentIdentity(
            claim=data.ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS,
        )
    return data.BenchmarkEvidenceBinding(
        binding_id=binding_id,
        physical_dataset_binding_id="synthetic-physical-dataset",
        dataset_id="synthetic-public-study",
        dataset_version="2026-08-09",
        manifest_artifact=artifact(
            f"{binding_id}-manifest",
            sha256=manifest_fingerprint,
        ),
        manifest_fingerprint=manifest_fingerprint,
        assessment_reference=assessment_reference(f"{binding_id}-assessment", manifest_fingerprint),
        assessment_kind=kind,
        assessment_identity=assessment_identity,
        scope_binding=data.EvidenceScopeBinding(
            assessment_scope_fingerprint=digest(f"{binding_id}-assessment-scope"),
            target_mappings=target_mappings,
            intervention_mappings=intervention_mappings,
            environment_mappings=environment_mappings,
            horizon_names=tuple(sorted(item.name for item in query.prediction_horizons)),
            scientific_claims=(data.ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS,),
        ),
        required_split_unit=unit(data.ExperimentalUnitLevel.WELL, "well_id"),
        representability_proof_fingerprint=digest(f"{binding_id}-proof"),
    )


def unit(level: data.ExperimentalUnitLevel, id_field: str) -> data.ExperimentalUnitBinding:
    return data.ExperimentalUnitBinding(
        level=level,
        identity=data.UnitIdentityExpression(
            kind=data.UnitIdentityExpressionKind.SOURCE_FIELD,
            source_fields=(id_field,),
        ),
    )


def id_membership(name: str, count: int) -> data.CanonicalIdMembership:
    return data.CanonicalIdMembership(
        ids_artifact=artifact(f"{name}-ids"),
        id_count=count,
    )


def exact_id_membership(name: str, ids: tuple[str, ...]) -> data.CanonicalIdMembership:
    canonical_ids = tuple(sorted(ids))
    encoded = canonical_json_bytes(canonical_ids)
    return data.CanonicalIdMembership(
        ids_artifact=artifact(
            f"{name}-ids",
            sha256=canonical_fingerprint(canonical_ids),
            byte_count=len(encoded),
        ),
        id_count=len(canonical_ids),
    )


def partition_well_ids(name: str, count: int) -> tuple[str, ...]:
    return tuple(f"{name}-well-{index:04d}" for index in range(count))


def explicit_membership(
    name: str,
    *,
    wells: int,
    cells: int,
) -> data.ExplicitPartitionMembership:
    well = unit(data.ExperimentalUnitLevel.WELL, "well_id")
    cell = unit(data.ExperimentalUnitLevel.CELL, "cell_id")
    well_membership = exact_id_membership(
        f"{name}-wells",
        partition_well_ids(name, wells),
    )
    return data.ExplicitPartitionMembership(
        assignment_unit=well,
        assignment_unit_ids=well_membership,
        record_unit=cell,
        record_ids=id_membership(f"{name}-cells", cells),
        descendant_closure_artifact=artifact(f"{name}-well-cell-closure"),
        protected_group_memberships=(
            data.ProtectedGroupMembership(
                unit=well,
                membership=well_membership,
            ),
        ),
    )


def split_plan(query: StateQuery) -> data.BenchmarkSplitPlan:
    well = unit(data.ExperimentalUnitLevel.WELL, "well_id")
    cell = unit(data.ExperimentalUnitLevel.CELL, "cell_id")
    universe = data.PartitionUniverse(
        physical_dataset_binding_id="synthetic-physical-dataset",
        slice_fingerprint=digest("synthetic-slice"),
        assignment_unit=well,
        assignment_unit_ids=id_membership("universe-wells", 8),
        record_unit=cell,
        record_ids=id_membership("universe-cells", 80),
        descendant_closure_artifact=artifact("universe-well-cell-closure"),
    )
    closure = data.ProtectedGroupClosure(
        physical_dataset_binding_id="synthetic-physical-dataset",
        unit_ancestry=(well, cell),
        record_unit=cell,
        assignment_unit=well,
        protected_groups=(
            data.ProtectedGroupBinding(
                unit=well,
                reasons=(
                    data.ProtectedGroupReason.METRIC_EVALUATION,
                    data.ProtectedGroupReason.OBJECTIVE_REQUIRED_SPLIT,
                    data.ProtectedGroupReason.SPLIT_ASSIGNMENT,
                ),
            ),
        ),
    )
    return data.BenchmarkSplitPlan(
        split_id="independent-well-split",
        split_version="1.0.0",
        query_fingerprint=query.fingerprint,
        universes=(universe,),
        protected_group_closures=(closure,),
        partitions=(
            data.BenchmarkPartition(
                partition_id="p1-train",
                role=data.BenchmarkPartitionRole.TRAIN,
                physical_dataset_binding_id="synthetic-physical-dataset",
                membership=explicit_membership("train", wells=4, cells=40),
            ),
            data.BenchmarkPartition(
                partition_id="p2-calibration",
                role=data.BenchmarkPartitionRole.CALIBRATION,
                physical_dataset_binding_id="synthetic-physical-dataset",
                membership=explicit_membership("calibration", wells=1, cells=10),
            ),
            data.BenchmarkPartition(
                partition_id="p3-validation",
                role=data.BenchmarkPartitionRole.MODEL_SELECTION_VALIDATION,
                physical_dataset_binding_id="synthetic-physical-dataset",
                membership=explicit_membership("validation", wells=1, cells=10),
            ),
            data.BenchmarkPartition(
                partition_id="p4-test",
                role=data.BenchmarkPartitionRole.UNTOUCHED_TEST,
                physical_dataset_binding_id="synthetic-physical-dataset",
                membership=explicit_membership("test", wells=2, cells=20),
            ),
        ),
    )


def evaluation_case_set(
    query: StateQuery,
    plan: data.BenchmarkSplitPlan,
) -> data.BenchmarkEvaluationCaseSet:
    context_fingerprint = digest("synthetic-static-context")
    context = data.EvaluationContextBinding(
        context_id="synthetic-static-context",
        context_fingerprint=context_fingerprint,
        context_artifact=artifact(
            "synthetic-static-context",
            sha256=context_fingerprint,
        ),
    )
    labels = {
        "p1-train": "train",
        "p2-calibration": "calibration",
        "p3-validation": "validation",
        "p4-test": "test",
    }
    action_ids = tuple(sorted(item.spec_id for item in query.intervention_space))
    cases: list[data.BenchmarkEvaluationCase] = []
    partition_bindings: list[data.EvaluationCasePartitionBinding] = []
    action_index = 0
    for partition in plan.partitions:
        membership = partition.materialized_membership
        assert membership is not None
        protected = next(
            item
            for item in membership.protected_group_memberships
            if item.unit.level is data.ExperimentalUnitLevel.WELL
        )
        label = labels[partition.partition_id]
        unit_ids = partition_well_ids(label, protected.membership.id_count)
        partition_bindings.append(
            data.EvaluationCasePartitionBinding(
                partition_id=partition.partition_id,
                evaluation_unit_ids=protected.membership,
            )
        )
        control_id = unit_ids[0]
        for index, unit_id in enumerate(unit_ids):
            treated = index > 0 and bool(action_ids)
            intervention_ids: tuple[str, ...] = ()
            matched_controls: tuple[str, ...] = ()
            role = data.EvaluationCaseRole.MATCHED_NO_ACTION_CONTROL
            if treated:
                intervention_ids = (action_ids[action_index % len(action_ids)],)
                action_index += 1
                matched_controls = (control_id,)
                role = data.EvaluationCaseRole.TREATED
            cases.append(
                data.BenchmarkEvaluationCase(
                    case_id=f"case-{unit_id}",
                    partition_id=partition.partition_id,
                    evaluation_unit_id=unit_id,
                    prediction_subject_id=unit_id,
                    context_id=context.context_id,
                    context_fingerprint=context.context_fingerprint,
                    matching_stratum_id="synthetic-stratum",
                    matching_stratum_fingerprint=digest("synthetic-stratum"),
                    intervention_spec_ids=intervention_ids,
                    horizon_name=query.prediction_horizons[0].name,
                    target_output_keys=tuple(
                        sorted(output.term.key for output in query.target_outputs)
                    ),
                    role=role,
                    matched_control_evaluation_unit_ids=matched_controls,
                )
            )
    cases_tuple = tuple(sorted(cases, key=lambda item: item.case_id))
    case_bytes = canonical_json_bytes([case.model_dump(mode="json") for case in cases_tuple])
    multiplicities = tuple(
        data.EvaluationInterventionMultiplicity(
            intervention_spec_id=action_id,
            case_count=sum(action_id in case.intervention_spec_ids for case in cases_tuple),
        )
        for action_id in action_ids
    )
    return data.BenchmarkEvaluationCaseSet(
        case_set_id="synthetic-exact-evaluation-cases",
        case_set_version="1.0.0",
        query_fingerprint=query.fingerprint,
        evaluation_unit=unit(data.ExperimentalUnitLevel.WELL, "well_id"),
        contexts=(context,),
        case_artifact=artifact(
            "synthetic-exact-evaluation-cases",
            sha256=canonical_fingerprint([case.model_dump(mode="json") for case in cases_tuple]),
            byte_count=len(case_bytes),
        ),
        case_count=len(cases_tuple),
        partition_memberships=tuple(partition_bindings),
        intervention_case_counts=multiplicities,
        no_action_control_case_count=sum(
            case.role is data.EvaluationCaseRole.MATCHED_NO_ACTION_CONTROL for case in cases_tuple
        ),
        cases=cases_tuple,
    )


def rebuild_case_set(
    old: data.BenchmarkEvaluationCaseSet,
    cases: tuple[data.BenchmarkEvaluationCase, ...],
) -> data.BenchmarkEvaluationCaseSet:
    ordered = tuple(sorted(cases, key=lambda item: item.case_id))
    case_payload = [case.model_dump(mode="json") for case in ordered]
    case_bytes = canonical_json_bytes(case_payload)
    partition_ids = tuple(item.partition_id for item in old.partition_memberships)
    memberships = tuple(
        data.EvaluationCasePartitionBinding(
            partition_id=partition_id,
            evaluation_unit_ids=exact_id_membership(
                f"rebuilt-{partition_id}",
                tuple(
                    case.evaluation_unit_id for case in ordered if case.partition_id == partition_id
                ),
            ),
        )
        for partition_id in partition_ids
    )
    action_ids = tuple(
        sorted({action_id for case in ordered for action_id in case.intervention_spec_ids})
    )
    return revalidate(
        data.BenchmarkEvaluationCaseSet,
        old,
        case_artifact=artifact(
            "rebuilt-case-set",
            sha256=canonical_fingerprint(case_payload),
            byte_count=len(case_bytes),
        ),
        case_count=len(ordered),
        partition_memberships=memberships,
        intervention_case_counts=tuple(
            data.EvaluationInterventionMultiplicity(
                intervention_spec_id=action_id,
                case_count=sum(action_id in case.intervention_spec_ids for case in ordered),
            )
            for action_id in action_ids
        ),
        no_action_control_case_count=sum(
            case.role is data.EvaluationCaseRole.MATCHED_NO_ACTION_CONTROL for case in ordered
        ),
        cases=ordered,
    )


def metric_definition(query: StateQuery) -> data.BenchmarkMetricDefinition:
    target = query.target_outputs[0]
    horizon = query.prediction_horizons[0]
    well = unit(data.ExperimentalUnitLevel.WELL, "well_id")
    return data.BenchmarkMetricDefinition(
        metric_id="well-weighted-proper-score",
        metric_version="1.0.0",
        family=data.MetricFamily.PREDICTIVE_PROPER_SCORE,
        query_fingerprint=query.fingerprint,
        evidence_binding_ids=("metric-evidence",),
        evaluation_partition_ids=("p4-test",),
        target_output_keys=(target.term.key,),
        horizon_names=(horizon.name,),
        implementation_binding=executable_binding("well-weighted-proper-score"),
        prediction_representation=data.PredictionRepresentation.DISTRIBUTION_PARAMETERS,
        target_representation=data.TargetRepresentation.CONTINUOUS,
        evaluation_unit=well,
        aggregation=data.MetricAggregation.MEAN,
        weighting=data.MetricWeightingPolicy(
            scheme=data.MetricWeightingScheme.EQUAL_EVALUATION_UNIT,
        ),
        direction=data.MetricDirection.MINIMIZE,
        units="nats_per_well",
        formula="mean_well(-log p(y_well))",
        missingness_policy=data.MetricMissingnessPolicy.ERROR_ON_MISSING,
        uncertainty=data.MetricUncertaintySpec(
            method=executable_binding("well-cluster-bootstrap"),
            resampling_scheme=data.MetricResamplingScheme.IID_EVALUATION_UNIT,
            dependence_units=(
                data.MetricDependenceUnit(
                    dependence_id="well",
                    kind=data.MetricDependenceKind.EXPERIMENTAL_UNIT,
                    identity=well.identity,
                    record_to_group_artifact=artifact("well-record-grouping"),
                    experimental_unit=well,
                ),
            ),
            confidence_level=0.95,
            resample_count=1000,
        ),
        minimum_evaluation_units=2,
    )


def metric_reference(metric: data.BenchmarkMetricDefinition) -> data.MetricDefinitionReference:
    return data.MetricDefinitionReference(
        metric_id=metric.metric_id,
        metric_version=metric.metric_version,
        metric_fingerprint=metric.fingerprint,
    )


def baseline_definition(query: StateQuery) -> data.BenchmarkBaselineDefinition:
    return data.BenchmarkBaselineDefinition(
        baseline_id="training-mean",
        baseline_version="1.0.0",
        query_fingerprint=query.fingerprint,
        implementation_binding=executable_binding("training-mean-baseline"),
        applicability=data.BaselineApplicabilityRule(
            allowed_subject_kinds=(query.subject.kind,),
            minimum_target_count=1,
            minimum_horizon_count=1,
        ),
        training_partition_ids=("p1-train",),
        seeds=(0,),
    )


def baseline_reference(
    baseline: data.BenchmarkBaselineDefinition,
) -> data.BaselineDefinitionReference:
    return data.BaselineDefinitionReference(
        baseline_id=baseline.baseline_id,
        baseline_version=baseline.baseline_version,
        baseline_fingerprint=baseline.fingerprint,
    )


def leakage_audit(plan: data.BenchmarkSplitPlan) -> data.BenchmarkLeakageAudit:
    partition_ids = tuple(partition.partition_id for partition in plan.partitions)
    pair_count = len(partition_ids) * (len(partition_ids) - 1) // 2
    well = unit(data.ExperimentalUnitLevel.WELL, "well_id")

    def passed_check(
        check_id: str,
        kind: data.LeakageCheckKind,
        *,
        evidence_id: str | None = None,
        protected_unit: data.ExperimentalUnitBinding | None = None,
        overlap: bool = False,
    ) -> data.LeakageAuditCheck:
        return data.LeakageAuditCheck(
            check_id=check_id,
            kind=kind,
            status=data.AuditCheckStatus.PASSED,
            physical_dataset_binding_id=evidence_id,
            protected_unit=protected_unit,
            partition_ids=partition_ids,
            comparisons_expected=pair_count if overlap else None,
            comparisons_completed=pair_count if overlap else None,
            overlap_count=0 if overlap else None,
            report_locator=f"checks/{check_id}",
            notes=("All required comparisons passed.",),
        )

    return data.BenchmarkLeakageAudit(
        audit_id="independent-unit-leakage-audit",
        audit_version="1.0.0",
        split_plan_fingerprint=plan.fingerprint,
        implementation=implementation("leakage-auditor"),
        report_artifact=artifact("leakage-audit-report"),
        evaluated_partition_ids=partition_ids,
        checks=(
            passed_check(
                "01-record-membership",
                data.LeakageCheckKind.RECORD_MEMBERSHIP_DISJOINT,
                evidence_id="synthetic-physical-dataset",
                overlap=True,
            ),
            passed_check(
                "02-protected-well",
                data.LeakageCheckKind.PROTECTED_GROUP_DISJOINT,
                evidence_id="synthetic-physical-dataset",
                protected_unit=well,
                overlap=True,
            ),
            passed_check(
                "03-source-duplicates",
                data.LeakageCheckKind.SOURCE_DUPLICATE_DISJOINT,
                overlap=True,
            ),
            passed_check(
                "04-preprocessing",
                data.LeakageCheckKind.PREPROCESSING_FIT_ISOLATED,
            ),
            passed_check(
                "05-target-derivation",
                data.LeakageCheckKind.TARGET_DERIVATION_ISOLATED,
            ),
            passed_check(
                "06-temporal-cutoff",
                data.LeakageCheckKind.TEMPORAL_CUTOFF_RESPECTED,
            ),
        ),
        reviewed_by=("benchmark-contract-test",),
        reviewed_on=date(2026, 8, 9),
    )


def admitted_artifact(query: StateQuery) -> data.BenchmarkArtifact:
    query_ref = query_binding(query)
    manifest_fingerprint = "a" * 64
    evidence = (
        evidence_binding(
            "loss-evidence",
            query=query,
            kind=data.AssessmentKind.LOSS,
            manifest_fingerprint=manifest_fingerprint,
        ),
        evidence_binding(
            "metric-evidence",
            query=query,
            kind=data.AssessmentKind.METRIC,
            manifest_fingerprint=manifest_fingerprint,
        ),
    )
    plan = split_plan(query)
    case_set = evaluation_case_set(query, plan)
    metric = metric_definition(query)
    baseline = baseline_definition(query)
    acceptance_rule = data.BenchmarkAcceptanceRule(
        rule_id="proper-score-test-threshold",
        metric=metric_reference(metric),
        partition_id="p4-test",
        comparison=data.ThresholdComparison.LESS_THAN_OR_EQUAL,
        estimate=data.ThresholdEstimate.POINT,
        absolute_threshold=1.0,
        rationale="Prespecified held-out proper-score threshold.",
    )
    definition = data.BenchmarkDefinition(
        benchmark_id="synthetic-independent-unit-benchmark",
        benchmark_version="1.0.0",
        title="Synthetic independent-unit benchmark",
        description="A strict contract fixture, not biological evidence.",
        design_status=data.BenchmarkLifecycle.FROZEN,
        intent=data.BenchmarkIntent.SCIENTIFIC,
        query=query_ref,
        scope=data.BenchmarkScope(
            scope_id="synthetic-scientific-scope",
            query_fingerprint=query.fingerprint,
            subject_kind=query.subject.kind,
            system_boundary=query.system_boundary,
            biological_system=query.subject.biological_system,
            target_output_keys=tuple(sorted(output.term.key for output in query.target_outputs)),
            horizon_names=tuple(sorted(item.name for item in query.prediction_horizons)),
            intervention_spec_ids=tuple(sorted(item.spec_id for item in query.intervention_space)),
            scientific_claims=(data.ScientificClaim.INDIVIDUAL_LONGITUDINAL_DYNAMICS,),
            inference_cutoff_seconds=0.0,
            reference_estimand_causal_status=CausalStatus.PREDICTIVE_ASSOCIATION,
            forecast_causal_status=CausalStatus.PREDICTIVE_ASSOCIATION,
            estimand="Future individual-cell functional capacity at the declared horizon.",
        ),
        evidence_bindings=evidence,
        split_plan=plan,
        evaluation_case_set=case_set,
        metrics=(metric,),
        baselines=(baseline,),
        acceptance_rules=(acceptance_rule,),
        acceptance_policy=data.BenchmarkAcceptancePolicy(
            policy_id="synthetic-acceptance-policy",
            policy_version="1.0.0",
            root_group_id="all-required",
            groups=(
                data.BenchmarkAcceptanceGroup(
                    group_id="all-required",
                    operator=data.AcceptanceGroupOperator.ALL,
                    rule_ids=(acceptance_rule.rule_id,),
                ),
            ),
        ),
    )
    audit = leakage_audit(plan)
    baseline_run = data.BaselineRun(
        baseline=baseline_reference(baseline),
        status=data.BaselineRunStatus.PASSED,
        applicability_rule_fingerprint=baseline.applicability.fingerprint,
        prediction_artifact=artifact("training-mean-test-predictions"),
        metric_results=(
            data.MetricResult(
                metric=metric_reference(metric),
                partition_id="p4-test",
                value=0.5,
                lower_confidence_bound=0.4,
                upper_confidence_bound=0.6,
                evaluated_evaluation_units=2,
                evaluated_case_ids=case_set.partition_memberships[-1].evaluation_unit_ids,
                result_artifact=artifact("training-mean-proper-score-result"),
            ),
        ),
        notes=("Baseline completed on the frozen split.",),
    )
    admission = data.BenchmarkAdmission(
        admission_id="synthetic-admission",
        admission_version="1.0.0",
        definition_fingerprint=definition.fingerprint,
        status=data.BenchmarkAdmissionStatus.ADMITTED,
        evidence_resolutions=tuple(
            data.EvidenceResolutionBinding(
                evidence_binding_id=binding.binding_id,
                assessment_reference=binding.assessment_reference,
                resolution_fingerprint=digest(f"{binding.binding_id}-resolution"),
            )
            for binding in evidence
        ),
        leakage_audit_fingerprint=audit.fingerprint,
        baseline_runs=(baseline_run,),
        metric_results=(
            data.MetricResult(
                metric=metric_reference(metric),
                partition_id="p4-test",
                value=0.4,
                lower_confidence_bound=0.3,
                upper_confidence_bound=0.5,
                evaluated_evaluation_units=2,
                evaluated_case_ids=case_set.partition_memberships[-1].evaluation_unit_ids,
                result_artifact=artifact("candidate-proper-score-result"),
            ),
        ),
        reasons=("All frozen structural admission gates passed.",),
        reviewed_by=("benchmark-contract-test",),
        reviewed_on=date(2026, 8, 9),
    )
    return data.BenchmarkArtifact(
        definition=definition,
        leakage_audit=audit,
        admission=admission,
    )


def revalidate(
    model_type: type[ModelT],
    model: BaseModel,
    **updates: object,
) -> ModelT:
    payload = model.model_dump(mode="python")
    payload.update(updates)
    return model_type.model_validate(payload)


def test_admitted_artifact_binds_query_evidence_splits_metrics_and_baseline(
    query: StateQuery,
) -> None:
    benchmark = admitted_artifact(query)
    assert benchmark.definition.query.state_query == query
    assert benchmark.definition.query.query_fingerprint == query.fingerprint
    assert benchmark.definition.query.query_artifact.sha256 == query.fingerprint
    assert benchmark.definition.design_status is data.BenchmarkLifecycle.FROZEN
    assert benchmark.admission.status is data.BenchmarkAdmissionStatus.ADMITTED
    assert benchmark.definition.split_plan is not None
    assert all(
        partition.materialized_membership is not None
        for partition in benchmark.definition.split_plan.partitions
    )
    assert benchmark.definition.metrics[0].aggregation is data.MetricAggregation.MEAN
    assert benchmark.definition.metrics[0].weighting.scheme is (
        data.MetricWeightingScheme.EQUAL_EVALUATION_UNIT
    )
    physical_ids = {
        binding.physical_dataset_binding_id for binding in benchmark.definition.evidence_bindings
    }
    assert physical_ids == {"synthetic-physical-dataset"}
    assert len(benchmark.definition.evidence_bindings) == 2
    assert len(benchmark.definition.split_plan.universes) == 1
    assert data.BenchmarkArtifact.model_validate_json(benchmark.model_dump_json()) == benchmark


def test_admission_cannot_omit_a_declared_metric_partition_result(
    query: StateQuery,
) -> None:
    benchmark = admitted_artifact(query)
    admission = revalidate(
        data.BenchmarkAdmission,
        benchmark.admission,
        metric_results=(),
    )
    with pytest.raises(ValidationError, match="every declared metric and evaluation partition"):
        data.BenchmarkArtifact(
            definition=benchmark.definition,
            leakage_audit=benchmark.leakage_audit,
            admission=admission,
        )


def test_metric_evaluation_partition_must_match_exact_assessment_purpose(
    query: StateQuery,
) -> None:
    benchmark = admitted_artifact(query)
    metric = revalidate(
        data.BenchmarkMetricDefinition,
        benchmark.definition.metrics[0],
        evaluation_partition_ids=("p3-validation",),
    )
    rule = revalidate(
        data.BenchmarkAcceptanceRule,
        benchmark.definition.acceptance_rules[0],
        metric=metric_reference(metric),
    )
    with pytest.raises(ValidationError, match="exact assessment purposes"):
        revalidate(
            data.BenchmarkDefinition,
            benchmark.definition,
            metrics=(metric,),
            acceptance_rules=(rule,),
        )


def test_evaluation_cases_fail_closed_on_missing_well_or_substituted_action(
    query: StateQuery,
) -> None:
    benchmark = admitted_artifact(query)
    case_set = benchmark.definition.evaluation_case_set
    assert case_set is not None
    test_case = next(
        case
        for case in case_set.cases
        if case.partition_id == "p4-test" and case.role is data.EvaluationCaseRole.TREATED
    )
    omitted = rebuild_case_set(
        case_set,
        tuple(case for case in case_set.cases if case.case_id != test_case.case_id),
    )
    with pytest.raises(ValidationError, match="exact partition evaluation-unit membership"):
        revalidate(
            data.BenchmarkDefinition,
            benchmark.definition,
            evaluation_case_set=omitted,
        )

    treated = next(case for case in case_set.cases if case.role is data.EvaluationCaseRole.TREATED)
    substituted = treated.model_copy(
        update={"intervention_spec_ids": ("unfrozen-intermediate-dose",)}
    )
    substituted_cases = tuple(
        substituted if case.case_id == treated.case_id else case for case in case_set.cases
    )
    substituted_set = rebuild_case_set(case_set, substituted_cases)
    with pytest.raises(ValidationError, match="exactly cover the query action domain"):
        revalidate(
            data.BenchmarkDefinition,
            benchmark.definition,
            evaluation_case_set=substituted_set,
        )


def test_evaluation_cases_require_same_stratum_explicit_no_action_controls(
    query: StateQuery,
) -> None:
    benchmark = admitted_artifact(query)
    case_set = benchmark.definition.evaluation_case_set
    assert case_set is not None
    treated = next(case for case in case_set.cases if case.role is data.EvaluationCaseRole.TREATED)
    control_id = treated.matched_control_evaluation_unit_ids[0]
    control = next(case for case in case_set.cases if case.evaluation_unit_id == control_id)
    drifted_control = control.model_copy(
        update={
            "matching_stratum_id": "wrong-stratum",
            "matching_stratum_fingerprint": digest("wrong-stratum"),
        }
    )
    drifted_cases = tuple(
        drifted_control if case.case_id == control.case_id else case for case in case_set.cases
    )
    with pytest.raises(ValidationError, match="share exact context and stratum"):
        rebuild_case_set(case_set, drifted_cases)


def test_query_binding_rejects_partial_payload_or_transport_drift(query: StateQuery) -> None:
    binding = query_binding(query)
    assert binding.parameter_grid is None
    with pytest.raises(ValidationError, match="byte count"):
        revalidate(
            data.StateQueryBinding,
            binding,
            query_artifact=binding.query_artifact.model_copy(
                update={"byte_count": binding.query_artifact.byte_count + 1}
            ),
        )
    changed_query = query.model_copy(update={"temporal_resolution_seconds": 2.0})
    with pytest.raises(ValidationError, match="fingerprint"):
        revalidate(data.StateQueryBinding, binding, state_query=changed_query)


def test_acceptance_tree_represents_all_noninferior_and_any_superior() -> None:
    policy = data.BenchmarkAcceptancePolicy(
        policy_id="noninferiority-and-superiority",
        policy_version="1.0.0",
        root_group_id="root-all",
        groups=(
            data.BenchmarkAcceptanceGroup(
                group_id="noninferior-all",
                operator=data.AcceptanceGroupOperator.ALL,
                rule_ids=("ni-a", "ni-b"),
            ),
            data.BenchmarkAcceptanceGroup(
                group_id="root-all",
                operator=data.AcceptanceGroupOperator.ALL,
                child_group_ids=("noninferior-all", "superior-any"),
            ),
            data.BenchmarkAcceptanceGroup(
                group_id="superior-any",
                operator=data.AcceptanceGroupOperator.ANY,
                rule_ids=("sup-a", "sup-b"),
            ),
        ),
    )
    assert policy.root_group_id == "root-all"
    with pytest.raises(ValidationError, match="mode, and requirement"):
        data.BenchmarkAcceptanceRule(
            rule_id="untyped-margin",
            metric=data.MetricDefinitionReference(
                metric_id="metric",
                metric_version="1",
                metric_fingerprint="a" * 64,
            ),
            partition_id="test",
            comparison=data.ThresholdComparison.LESS_THAN_OR_EQUAL,
            estimate=data.ThresholdEstimate.POINT,
            baseline_margin=0.1,
            rationale="A margin without typed semantics must fail.",
        )


def test_baseline_relative_gate_requires_paired_blocked_comparison(
    query: StateQuery,
) -> None:
    benchmark = admitted_artifact(query)
    metric = benchmark.definition.metrics[0]
    baseline = benchmark.definition.baselines[0]
    paired_rule = revalidate(
        data.BenchmarkAcceptanceRule,
        benchmark.definition.acceptance_rules[0],
        estimate=data.ThresholdEstimate.UPPER_CONFIDENCE_BOUND,
        absolute_threshold=None,
        baseline_margin=0.1,
        baseline_comparator=data.ExactBaselineComparator(baseline=baseline_reference(baseline)),
        baseline_margin_mode=data.BaselineMarginMode.ABSOLUTE_DIFFERENCE,
        baseline_requirement=data.BaselineRequirement.NONINFERIOR,
        confidence_level=0.95,
    )
    definition = revalidate(
        data.BenchmarkDefinition,
        benchmark.definition,
        acceptance_rules=(paired_rule,),
    )
    admission_without_pairing = revalidate(
        data.BenchmarkAdmission,
        benchmark.admission,
        definition_fingerprint=definition.fingerprint,
    )
    with pytest.raises(ValidationError, match="do not pass the frozen acceptance policy"):
        data.BenchmarkArtifact(
            definition=definition,
            leakage_audit=benchmark.leakage_audit,
            admission=admission_without_pairing,
        )

    case_set = definition.evaluation_case_set
    assert case_set is not None
    paired = data.PairedBaselineComparisonResult(
        comparison_id="proper-score-vs-training-mean",
        metric=metric_reference(metric),
        partition_id="p4-test",
        baseline=baseline_reference(baseline),
        effect_scale=data.BaselineMarginMode.ABSOLUTE_DIFFERENCE,
        effect_definition="candidate_minus_baseline_v1",
        point_effect=-0.1,
        one_sided_confidence_bound=-0.05,
        bound_kind=data.PairedConfidenceBoundKind.UPPER,
        confidence_level=0.95,
        evaluated_case_ids=case_set.partition_memberships[-1].evaluation_unit_ids,
        dependence_ids=("well",),
        paired_block_membership_artifact=artifact("paired-well-block-membership"),
        result_artifact=artifact("paired-proper-score-comparison"),
    )
    admission = revalidate(
        data.BenchmarkAdmission,
        admission_without_pairing,
        paired_baseline_comparisons=(paired,),
    )
    accepted = data.BenchmarkArtifact(
        definition=definition,
        leakage_audit=benchmark.leakage_audit,
        admission=admission,
    )
    assert accepted.admission.paired_baseline_comparisons == (paired,)


def test_relative_paired_effect_rejects_zero_baseline_denominator(
    query: StateQuery,
) -> None:
    benchmark = admitted_artifact(query)
    baseline_run = benchmark.admission.baseline_runs[0]
    zero_result = baseline_run.metric_results[0].model_copy(
        update={
            "value": 0.0,
            "lower_confidence_bound": -0.1,
            "upper_confidence_bound": 0.1,
        }
    )
    zero_run = revalidate(
        data.BaselineRun,
        baseline_run,
        metric_results=(zero_result,),
    )
    case_set = benchmark.definition.evaluation_case_set
    assert case_set is not None
    comparison = data.PairedBaselineComparisonResult(
        comparison_id="undefined-relative-comparison",
        metric=metric_reference(benchmark.definition.metrics[0]),
        partition_id="p4-test",
        baseline=baseline_reference(benchmark.definition.baselines[0]),
        effect_scale=data.BaselineMarginMode.RELATIVE_FRACTION,
        effect_definition="candidate_minus_baseline_over_abs_baseline_v1",
        point_effect=0.0,
        one_sided_confidence_bound=0.1,
        bound_kind=data.PairedConfidenceBoundKind.UPPER,
        confidence_level=0.95,
        evaluated_case_ids=case_set.partition_memberships[-1].evaluation_unit_ids,
        dependence_ids=("well",),
        paired_block_membership_artifact=artifact("relative-paired-blocks"),
        result_artifact=artifact("undefined-relative-result"),
    )
    admission = revalidate(
        data.BenchmarkAdmission,
        benchmark.admission,
        baseline_runs=(zero_run,),
        paired_baseline_comparisons=(comparison,),
    )
    with pytest.raises(ValidationError, match="undefined for a zero baseline"):
        data.BenchmarkArtifact(
            definition=benchmark.definition,
            leakage_audit=benchmark.leakage_audit,
            admission=admission,
        )


def test_split_assignment_cannot_be_finer_than_protected_parent() -> None:
    well = unit(data.ExperimentalUnitLevel.WELL, "well_id")
    cell = unit(data.ExperimentalUnitLevel.CELL, "cell_id")
    with pytest.raises(ValidationError, match="finer than a protected group"):
        data.ProtectedGroupClosure(
            physical_dataset_binding_id="synthetic-physical-dataset",
            unit_ancestry=(well, cell),
            record_unit=cell,
            assignment_unit=cell,
            protected_groups=(
                data.ProtectedGroupBinding(
                    unit=cell,
                    reasons=(data.ProtectedGroupReason.SPLIT_ASSIGNMENT,),
                ),
                data.ProtectedGroupBinding(
                    unit=well,
                    reasons=(data.ProtectedGroupReason.BIOLOGICAL_REPLICATE,),
                ),
            ),
        )


def test_plate_assignment_can_score_wells_with_plate_and_compound_clustering(
    query: StateQuery,
) -> None:
    benchmark = admitted_artifact(query)
    old_plan = benchmark.definition.split_plan
    assert old_plan is not None
    plate = unit(data.ExperimentalUnitLevel.PLATE, "plate_id")
    well = unit(data.ExperimentalUnitLevel.WELL, "well_id")
    cell = unit(data.ExperimentalUnitLevel.CELL, "cell_id")

    def membership(
        name: str,
        *,
        plate_count: int,
        well_count: int,
        cell_count: int,
    ) -> data.ExplicitPartitionMembership:
        well_membership = exact_id_membership(
            f"{name}-wells",
            partition_well_ids(name, well_count),
        )
        return data.ExplicitPartitionMembership(
            assignment_unit=plate,
            assignment_unit_ids=id_membership(f"{name}-plates", plate_count),
            record_unit=cell,
            record_ids=id_membership(f"{name}-cells", cell_count),
            descendant_closure_artifact=artifact(f"{name}-plate-well-cell-closure"),
            protected_group_memberships=(
                data.ProtectedGroupMembership(
                    unit=plate,
                    membership=id_membership(f"{name}-protected-plates", plate_count),
                ),
                data.ProtectedGroupMembership(
                    unit=well,
                    membership=well_membership,
                ),
            ),
        )

    plan = data.BenchmarkSplitPlan(
        split_id="whole-plate-split-with-well-evaluation",
        split_version="1.0.0",
        query_fingerprint=query.fingerprint,
        universes=(
            data.PartitionUniverse(
                physical_dataset_binding_id="synthetic-physical-dataset",
                slice_fingerprint=digest("plate-slice"),
                assignment_unit=plate,
                assignment_unit_ids=id_membership("universe-plates", 4),
                record_unit=cell,
                record_ids=id_membership("plate-universe-cells", 80),
                descendant_closure_artifact=artifact("universe-plate-well-cell-closure"),
            ),
        ),
        protected_group_closures=(
            data.ProtectedGroupClosure(
                physical_dataset_binding_id="synthetic-physical-dataset",
                unit_ancestry=(plate, well, cell),
                record_unit=cell,
                assignment_unit=plate,
                protected_groups=(
                    data.ProtectedGroupBinding(
                        unit=plate,
                        reasons=(
                            data.ProtectedGroupReason.OBJECTIVE_REQUIRED_SPLIT,
                            data.ProtectedGroupReason.SPLIT_ASSIGNMENT,
                        ),
                    ),
                    data.ProtectedGroupBinding(
                        unit=well,
                        reasons=(
                            data.ProtectedGroupReason.METRIC_EVALUATION,
                            data.ProtectedGroupReason.OBJECTIVE_REQUIRED_SPLIT,
                        ),
                    ),
                ),
            ),
        ),
        partitions=tuple(
            data.BenchmarkPartition(
                partition_id=partition_id,
                role=role,
                physical_dataset_binding_id="synthetic-physical-dataset",
                membership=membership(
                    label,
                    plate_count=1,
                    well_count=wells,
                    cell_count=cells,
                ),
            )
            for partition_id, role, label, wells, cells in (
                ("p1-train", data.BenchmarkPartitionRole.TRAIN, "train", 4, 40),
                (
                    "p2-calibration",
                    data.BenchmarkPartitionRole.CALIBRATION,
                    "calibration",
                    1,
                    10,
                ),
                (
                    "p3-validation",
                    data.BenchmarkPartitionRole.MODEL_SELECTION_VALIDATION,
                    "validation",
                    1,
                    10,
                ),
                (
                    "p4-test",
                    data.BenchmarkPartitionRole.UNTOUCHED_TEST,
                    "test",
                    2,
                    20,
                ),
            )
        ),
    )
    old_metric = benchmark.definition.metrics[0]
    compound_identity = data.UnitIdentityExpression(
        kind=data.UnitIdentityExpressionKind.SOURCE_FIELD,
        source_fields=("compound_dose_id",),
    )
    metric = revalidate(
        data.BenchmarkMetricDefinition,
        old_metric,
        evaluation_unit=well,
        weighting=data.MetricWeightingPolicy(
            scheme=data.MetricWeightingScheme.EQUAL_GROUP_THEN_EQUAL_EVALUATION_UNIT,
            group_dependence_id="compound-dose",
        ),
        uncertainty=data.MetricUncertaintySpec(
            method=executable_binding("multiway-cluster-bootstrap"),
            resampling_scheme=data.MetricResamplingScheme.MULTIWAY_CLUSTERED,
            dependence_units=(
                data.MetricDependenceUnit(
                    dependence_id="compound-dose",
                    kind=data.MetricDependenceKind.INTERVENTION_CONDITION,
                    identity=compound_identity,
                    record_to_group_artifact=artifact("compound-dose-record-grouping"),
                ),
                data.MetricDependenceUnit(
                    dependence_id="plate",
                    kind=data.MetricDependenceKind.EXPERIMENTAL_UNIT,
                    identity=plate.identity,
                    record_to_group_artifact=artifact("plate-record-grouping"),
                    experimental_unit=plate,
                ),
            ),
            confidence_level=0.95,
            resample_count=1000,
        ),
    )
    evidence = tuple(
        revalidate(
            data.BenchmarkEvidenceBinding,
            binding,
            required_split_unit=plate,
        )
        if binding.binding_id == "metric-evidence"
        else binding
        for binding in benchmark.definition.evidence_bindings
    )
    rule = revalidate(
        data.BenchmarkAcceptanceRule,
        benchmark.definition.acceptance_rules[0],
        metric=metric_reference(metric),
    )
    definition = revalidate(
        data.BenchmarkDefinition,
        benchmark.definition,
        evidence_bindings=evidence,
        split_plan=plan,
        evaluation_case_set=evaluation_case_set(query, plan),
        metrics=(metric,),
        acceptance_rules=(rule,),
    )
    assert definition.split_plan is not None
    assert definition.split_plan.protected_group_closures[0].assignment_unit == plate
    assert definition.metrics[0].evaluation_unit == well

    record_metric = revalidate(
        data.BenchmarkMetricDefinition,
        metric,
        evaluation_unit=cell,
    )
    record_rule = revalidate(
        data.BenchmarkAcceptanceRule,
        rule,
        metric=metric_reference(record_metric),
    )
    with pytest.raises(ValidationError, match="authoritative case evaluation unit"):
        revalidate(
            data.BenchmarkDefinition,
            definition,
            metrics=(record_metric,),
            acceptance_rules=(record_rule,),
        )


def test_seed_and_generator_without_materialized_membership_cannot_be_admitted(
    query: StateQuery,
) -> None:
    benchmark = admitted_artifact(query)
    plan = benchmark.definition.split_plan
    assert plan is not None
    generated = data.PartitionGenerationSpec(
        generator=implementation("well-split-generator"),
        source_universe_fingerprint=plan.universes[0].fingerprint,
        assignment_unit=unit(data.ExperimentalUnitLevel.WELL, "well_id"),
        seed=7,
    )
    partitions = (
        plan.partitions[0].model_copy(update={"membership": generated}),
        *plan.partitions[1:],
    )
    generated_plan = revalidate(data.BenchmarkSplitPlan, plan, partitions=partitions)
    with pytest.raises(ValidationError, match="require materialized partition membership"):
        revalidate(
            data.BenchmarkDefinition,
            benchmark.definition,
            split_plan=generated_plan,
        )


def test_record_pooled_metric_cannot_claim_scientific_admission(query: StateQuery) -> None:
    benchmark = admitted_artifact(query)
    old_metric = benchmark.definition.metrics[0]
    pooled_metric = revalidate(
        data.BenchmarkMetricDefinition,
        old_metric,
        aggregation=data.MetricAggregation.POOLED_RECORD,
        weighting=data.MetricWeightingPolicy(
            scheme=data.MetricWeightingScheme.RECORD_COUNT_WEIGHTED,
        ),
    )
    old_rule = benchmark.definition.acceptance_rules[0]
    rule = revalidate(
        data.BenchmarkAcceptanceRule,
        old_rule,
        metric=metric_reference(pooled_metric),
    )
    definition = revalidate(
        data.BenchmarkDefinition,
        benchmark.definition,
        metrics=(pooled_metric,),
        acceptance_rules=(rule,),
    )
    old_run = benchmark.admission.baseline_runs[0]
    result = old_run.metric_results[0].model_copy(
        update={"metric": metric_reference(pooled_metric)}
    )
    run = revalidate(data.BaselineRun, old_run, metric_results=(result,))
    admission = revalidate(
        data.BenchmarkAdmission,
        benchmark.admission,
        definition_fingerprint=definition.fingerprint,
        baseline_runs=(run,),
        metric_results=(
            benchmark.admission.metric_results[0].model_copy(
                update={"metric": metric_reference(pooled_metric)}
            ),
        ),
    )
    with pytest.raises(ValidationError, match="must not weight by record or cell count"):
        data.BenchmarkArtifact(
            definition=definition,
            leakage_audit=benchmark.leakage_audit,
            admission=admission,
        )


def test_metric_cannot_borrow_an_assessment_with_another_id_or_family(
    query: StateQuery,
) -> None:
    benchmark = admitted_artifact(query)
    old_metric = benchmark.definition.metrics[0]
    calibration_metric = revalidate(
        data.BenchmarkMetricDefinition,
        old_metric,
        family=data.MetricFamily.CALIBRATION,
    )
    with pytest.raises(ValidationError, match="exact assessment metric ID and family"):
        revalidate(
            data.BenchmarkDefinition,
            benchmark.definition,
            metrics=(calibration_metric,),
        )

    renamed_metric = revalidate(
        data.BenchmarkMetricDefinition,
        old_metric,
        metric_id="another-proper-score",
    )
    with pytest.raises(ValidationError, match="exact assessment metric ID and family"):
        revalidate(
            data.BenchmarkDefinition,
            benchmark.definition,
            metrics=(renamed_metric,),
        )


def test_failed_leakage_check_or_baseline_crash_blocks_admission(query: StateQuery) -> None:
    benchmark = admitted_artifact(query)
    assert benchmark.leakage_audit is not None
    old_check = benchmark.leakage_audit.checks[0]
    failed_check = revalidate(
        data.LeakageAuditCheck,
        old_check,
        status=data.AuditCheckStatus.FAILED,
        overlap_count=1,
        notes=(),
        blockers=("One record was shared across partitions.",),
    )
    failed_audit = revalidate(
        data.BenchmarkLeakageAudit,
        benchmark.leakage_audit,
        checks=(failed_check, *benchmark.leakage_audit.checks[1:]),
    )
    admission = revalidate(
        data.BenchmarkAdmission,
        benchmark.admission,
        leakage_audit_fingerprint=failed_audit.fingerprint,
    )
    with pytest.raises(ValidationError, match="leakage checks block admission"):
        data.BenchmarkArtifact(
            definition=benchmark.definition,
            leakage_audit=failed_audit,
            admission=admission,
        )

    old_run = benchmark.admission.baseline_runs[0]
    crashed = revalidate(
        data.BaselineRun,
        old_run,
        status=data.BaselineRunStatus.CRASHED,
        prediction_artifact=None,
        metric_results=(),
        notes=(),
        blockers=("Baseline process exited nonzero.",),
    )
    crash_admission = revalidate(
        data.BenchmarkAdmission,
        benchmark.admission,
        baseline_runs=(crashed,),
    )
    with pytest.raises(ValidationError, match="baselines block admission"):
        data.BenchmarkArtifact(
            definition=benchmark.definition,
            leakage_audit=benchmark.leakage_audit,
            admission=crash_admission,
        )


def test_baseline_not_applicable_is_machine_derived_from_exact_query(query: StateQuery) -> None:
    benchmark = admitted_artifact(query)
    old_run = benchmark.admission.baseline_runs[0]
    false_na = revalidate(
        data.BaselineRun,
        old_run,
        status=data.BaselineRunStatus.NOT_APPLICABLE,
        prediction_artifact=None,
        metric_results=(),
        notes=("Claimed not applicable.",),
        blockers=(),
    )
    admission = revalidate(
        data.BenchmarkAdmission,
        benchmark.admission,
        baseline_runs=(false_na,),
    )
    with pytest.raises(ValidationError, match="N/A status must be derived"):
        data.BenchmarkArtifact(
            definition=benchmark.definition,
            leakage_audit=benchmark.leakage_audit,
            admission=admission,
        )


def test_pre_cutoff_baseline_applicability_requires_count_and_target_modality_overlap(
    query: StateQuery,
) -> None:
    rule = data.BaselineApplicabilityRule(
        allowed_subject_kinds=(query.subject.kind,),
        requires_pre_cutoff_target_observation=True,
        minimum_target_count=1,
        minimum_horizon_count=1,
    )
    assert rule.applies_to(query) is False

    target_modality_without_count = query.evidence_policy.model_copy(
        update={"allowed_modalities": (query.target_outputs[0].term,)}
    )
    no_count_query = StateQuery.model_validate(
        {
            **query.model_dump(mode="python"),
            "evidence_policy": target_modality_without_count,
        }
    )
    assert rule.applies_to(no_count_query) is False

    count_without_target = query.evidence_policy.model_copy(
        update={"minimum_observed_measurements": 1}
    )
    no_overlap_query = StateQuery.model_validate(
        {
            **query.model_dump(mode="python"),
            "evidence_policy": count_without_target,
        }
    )
    assert rule.applies_to(no_overlap_query) is False

    observed_target = count_without_target.model_copy(
        update={"allowed_modalities": (query.target_outputs[0].term,)}
    )
    applicable_query = StateQuery.model_validate(
        {
            **query.model_dump(mode="python"),
            "evidence_policy": observed_target,
        }
    )
    assert rule.applies_to(applicable_query) is True


def test_complete_technical_only_artifact_needs_no_numeric_sentinels(query: StateQuery) -> None:
    admitted = admitted_artifact(query)
    scope = revalidate(
        data.BenchmarkScope,
        admitted.definition.scope,
        scientific_claims=(),
        reference_estimand_causal_status=CausalStatus.UNSUPPORTED,
        forecast_causal_status=CausalStatus.UNSUPPORTED,
    )
    definition = revalidate(
        data.BenchmarkDefinition,
        admitted.definition,
        intent=data.BenchmarkIntent.TECHNICAL_FIXTURE,
        scope=scope,
        split_plan=None,
        evaluation_case_set=None,
        metrics=(),
        baselines=(),
        acceptance_rules=(),
        acceptance_policy=None,
    )
    admission = revalidate(
        data.BenchmarkAdmission,
        admitted.admission,
        definition_fingerprint=definition.fingerprint,
        status=data.BenchmarkAdmissionStatus.TECHNICAL_ONLY,
        leakage_audit_fingerprint=None,
        baseline_runs=(),
        metric_results=(),
        reasons=("Parser and schema fixture only; no biological performance claim.",),
    )
    technical = data.BenchmarkArtifact(
        definition=definition,
        leakage_audit=None,
        admission=admission,
    )
    assert technical.admission.status is data.BenchmarkAdmissionStatus.TECHNICAL_ONLY
    assert technical.definition.split_plan is None
    assert technical.definition.acceptance_rules == ()

    scientific_definition = revalidate(
        data.BenchmarkDefinition,
        definition,
        intent=data.BenchmarkIntent.SCIENTIFIC,
        scope=admitted.definition.scope,
    )
    scientific_admission = revalidate(
        data.BenchmarkAdmission,
        admission,
        definition_fingerprint=scientific_definition.fingerprint,
    )
    with pytest.raises(ValidationError, match="explicit technical-fixture intent"):
        data.BenchmarkArtifact(
            definition=scientific_definition,
            leakage_audit=None,
            admission=scientific_admission,
        )


def test_frozen_component_benchmark_permits_not_run_baselines(query: StateQuery) -> None:
    admitted = admitted_artifact(query)
    definition = revalidate(
        data.BenchmarkDefinition,
        admitted.definition,
        intent=data.BenchmarkIntent.COMPONENT_BENCHMARK,
    )
    not_run = revalidate(
        data.BaselineRun,
        admitted.admission.baseline_runs[0],
        status=data.BaselineRunStatus.NOT_RUN,
        prediction_artifact=None,
        metric_results=(),
        notes=(),
        blockers=("Baseline execution remains an admission blocker.",),
    )
    admission = revalidate(
        data.BenchmarkAdmission,
        admitted.admission,
        definition_fingerprint=definition.fingerprint,
        status=data.BenchmarkAdmissionStatus.COMPONENT_BENCHMARK,
        baseline_runs=(not_run,),
        reasons=("Frozen component design; scientific admission was not claimed.",),
    )
    component = data.BenchmarkArtifact(
        definition=definition,
        leakage_audit=admitted.leakage_audit,
        admission=admission,
    )
    assert component.definition.design_status is data.BenchmarkLifecycle.FROZEN
    assert component.admission.status is data.BenchmarkAdmissionStatus.COMPONENT_BENCHMARK
    assert component.admission.baseline_runs[0].status is data.BaselineRunStatus.NOT_RUN


def test_component_may_freeze_specs_but_scientific_admission_requires_executables(
    query: StateQuery,
) -> None:
    admitted = admitted_artifact(query)
    old_metric = admitted.definition.metrics[0]
    planned_metric = revalidate(
        data.BenchmarkMetricDefinition,
        old_metric,
        implementation_binding=planned_binding("planned-metric"),
        uncertainty=old_metric.uncertainty.model_copy(
            update={"method": planned_binding("planned-uncertainty")}
        ),
    )
    old_baseline = admitted.definition.baselines[0]
    planned_baseline = revalidate(
        data.BenchmarkBaselineDefinition,
        old_baseline,
        implementation_binding=planned_binding("planned-baseline"),
    )
    rule = revalidate(
        data.BenchmarkAcceptanceRule,
        admitted.definition.acceptance_rules[0],
        metric=metric_reference(planned_metric),
    )
    component_definition = revalidate(
        data.BenchmarkDefinition,
        admitted.definition,
        intent=data.BenchmarkIntent.COMPONENT_BENCHMARK,
        metrics=(planned_metric,),
        baselines=(planned_baseline,),
        acceptance_rules=(rule,),
    )
    assert isinstance(
        component_definition.metrics[0].implementation_binding,
        data.SpecificationOnlyImplementationBinding,
    )

    old_run = admitted.admission.baseline_runs[0]
    planned_run = revalidate(
        data.BaselineRun,
        old_run,
        baseline=baseline_reference(planned_baseline),
        metric_results=tuple(
            result.model_copy(update={"metric": metric_reference(planned_metric)})
            for result in old_run.metric_results
        ),
    )
    scientific_definition = revalidate(
        data.BenchmarkDefinition,
        component_definition,
        intent=data.BenchmarkIntent.SCIENTIFIC,
    )
    admission = revalidate(
        data.BenchmarkAdmission,
        admitted.admission,
        definition_fingerprint=scientific_definition.fingerprint,
        baseline_runs=(planned_run,),
        metric_results=tuple(
            result.model_copy(update={"metric": metric_reference(planned_metric)})
            for result in admitted.admission.metric_results
        ),
    )
    with pytest.raises(ValidationError, match="requires executable metric and baseline"):
        data.BenchmarkArtifact(
            definition=scientific_definition,
            leakage_audit=admitted.leakage_audit,
            admission=admission,
        )


def test_exact_scope_mapping_rejects_generic_or_wrong_query_borrowing(
    query: StateQuery,
) -> None:
    admitted = admitted_artifact(query)
    metric_binding = admitted.definition.evidence_bindings[1]
    incomplete_scope = revalidate(
        data.EvidenceScopeBinding,
        metric_binding.scope_binding,
        intervention_mappings=(),
    )
    incomplete_binding = revalidate(
        data.BenchmarkEvidenceBinding,
        metric_binding,
        scope_binding=incomplete_scope,
    )
    with pytest.raises(ValidationError, match="map every exact intervention spec"):
        revalidate(
            data.BenchmarkDefinition,
            admitted.definition,
            evidence_bindings=(admitted.definition.evidence_bindings[0], incomplete_binding),
        )

    target_mapping = metric_binding.scope_binding.target_mappings[0]
    wrong_target = revalidate(
        data.EvidenceTargetMapping,
        target_mapping,
        target_output_fingerprint="0" * 64,
    )
    wrong_scope = revalidate(
        data.EvidenceScopeBinding,
        metric_binding.scope_binding,
        target_mappings=(wrong_target,),
    )
    wrong_binding = revalidate(
        data.BenchmarkEvidenceBinding,
        metric_binding,
        scope_binding=wrong_scope,
    )
    with pytest.raises(ValidationError, match="exact output spec"):
        revalidate(
            data.BenchmarkDefinition,
            admitted.definition,
            evidence_bindings=(admitted.definition.evidence_bindings[0], wrong_binding),
        )

    with pytest.raises(ValidationError, match="identified-effect status requires"):
        revalidate(
            data.BenchmarkScope,
            admitted.definition.scope,
            reference_estimand_causal_status=CausalStatus.IDENTIFIED_POPULATION_EFFECT,
        )


def population_query(base: StateQuery) -> StateQuery:
    subject = SubjectSpecification(
        kind=SubjectKind.POPULATION,
        biological_system=OntologyTerm(label="cultured cell population"),
        membership_semantics="one declared experimental population",
        experimental_unit_kind="sample",
        allowed_identity_bases=(IdentityBasis.EXPERIMENTAL_UNIT,),
    )
    horizon = PredictionHorizon(
        name="24h",
        duration_seconds=86_400.0,
        timescale=Timescale.SLOW,
    )
    target = base.target_outputs[0].model_copy(
        update={
            "aggregation": TargetAggregation(
                subject_kind=SubjectKind.POPULATION,
                statistic=AggregationStatistic.DISTRIBUTION,
                experimental_unit="sample",
            ),
            "supported_horizon_names": ("24h",),
        }
    )
    return StateQuery.model_validate(
        {
            **base.model_dump(mode="python"),
            "subject": subject,
            "system_boundary": SystemBoundary.POPULATION,
            "prediction_horizons": (horizon,),
            "target_outputs": (target,),
            "intervention_space": (),
        }
    )


def blocked_evidence_artifact(
    query: StateQuery,
    *,
    use_metric_assessment: bool = False,
    use_loss_assessment: bool = False,
) -> tuple[data.BenchmarkArtifact, data.DatasetManifest]:
    if use_metric_assessment and use_loss_assessment:
        raise ValueError("test helper accepts at most one objective kind")
    manifest = manifest_factory()
    claim = manifest.claim_assessments[0]
    assessment = (
        manifest.metric_assessments[0]
        if use_metric_assessment
        else manifest.loss_assessments[0]
        if use_loss_assessment
        else claim
    )
    reference = data.DatasetAssessmentReference(
        dataset_manifest_fingerprint=manifest.fingerprint,
        assessment_id=assessment.assessment_id,
        assessment_fingerprint=assessment.fingerprint,
    )
    manifest_bytes = manifest.canonical_json_bytes
    binding_id = (
        "metric-evidence"
        if use_metric_assessment
        else "loss-evidence"
        if use_loss_assessment
        else "claim-evidence"
    )
    assessment_kind = (
        data.AssessmentKind.METRIC
        if use_metric_assessment
        else data.AssessmentKind.LOSS
        if use_loss_assessment
        else data.AssessmentKind.CLAIM
    )
    assessment_identity: data.AssessmentIdentity = (
        data.MetricAssessmentIdentity(
            metric_id=manifest.metric_assessments[0].metric_id,
            metric_family=manifest.metric_assessments[0].metric_family,
            partition_purpose=manifest.metric_assessments[0].partition_purpose,
        )
        if use_metric_assessment
        else data.LossAssessmentIdentity(
            loss_kind=manifest.loss_assessments[0].loss_kind,
        )
        if use_loss_assessment
        else data.ClaimAssessmentIdentity(claim=claim.claim)
    )
    evidence = data.BenchmarkEvidenceBinding(
        binding_id=binding_id,
        physical_dataset_binding_id="blocked-public-study-physical",
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.version,
        manifest_artifact=artifact(
            "v03-canonical-manifest",
            sha256=manifest.fingerprint,
            byte_count=len(manifest_bytes),
        ),
        manifest_fingerprint=manifest.fingerprint,
        assessment_reference=reference,
        assessment_kind=assessment_kind,
        assessment_identity=assessment_identity,
        scope_binding=data.EvidenceScopeBinding(
            assessment_scope_fingerprint=assessment.scope.fingerprint,
            target_mappings=(
                data.EvidenceTargetMapping(
                    target_output_key=query.target_outputs[0].term.key,
                    target_output_fingerprint=canonical_fingerprint(
                        query.target_outputs[0].model_dump(mode="json")
                    ),
                    target_units=query.target_outputs[0].units,
                    target_aggregation=query.target_outputs[0].aggregation,
                    aggregation_unit=unit(
                        data.ExperimentalUnitLevel.SAMPLE,
                        "sample_id",
                    ),
                    assessment_modalities=tuple(
                        sorted(term.key for term in assessment.scope.modalities)
                    ),
                    semantics_artifact=artifact("population-target-semantics"),
                ),
            ),
            horizon_names=("24h",),
            scientific_claims=(data.ScientificClaim.POPULATION_DYNAMICS,),
        ),
        required_split_unit=unit(data.ExperimentalUnitLevel.WELL, "well_id"),
        representability_proof_fingerprint=digest("v03-representability-proof"),
    )
    definition = data.BenchmarkDefinition(
        benchmark_id="blocked-population-design",
        benchmark_version="1.0.0",
        title="Blocked population benchmark design",
        description="Frozen query and evidence binding without an admitted split.",
        design_status=data.BenchmarkLifecycle.FROZEN,
        intent=data.BenchmarkIntent.SCIENTIFIC,
        query=query_binding(query),
        scope=data.BenchmarkScope(
            scope_id="blocked-population-scope",
            query_fingerprint=query.fingerprint,
            subject_kind=query.subject.kind,
            system_boundary=query.system_boundary,
            biological_system=query.subject.biological_system,
            target_output_keys=tuple(sorted(output.term.key for output in query.target_outputs)),
            horizon_names=("24h",),
            intervention_spec_ids=tuple(sorted(item.spec_id for item in query.intervention_space)),
            scientific_claims=(data.ScientificClaim.POPULATION_DYNAMICS,),
            inference_cutoff_seconds=0.0,
            reference_estimand_causal_status=CausalStatus.PREDICTIVE_ASSOCIATION,
            forecast_causal_status=CausalStatus.PREDICTIVE_ASSOCIATION,
            estimand="Future population distribution at 24 hours.",
        ),
        evidence_bindings=(evidence,),
    )
    resolution = manifest.resolve_assessment(
        reference,
        use_case=data.DataUseCase.BENCHMARK_EVALUATION,
    )
    admission = data.BenchmarkAdmission(
        admission_id="blocked-population-admission",
        admission_version="1.0.0",
        definition_fingerprint=definition.fingerprint,
        status=data.BenchmarkAdmissionStatus.BLOCKED,
        evidence_resolutions=(
            data.EvidenceResolutionBinding(
                evidence_binding_id=evidence.binding_id,
                assessment_reference=reference,
                resolution_fingerprint=canonical_fingerprint(resolution.model_dump(mode="json")),
            ),
        ),
        reasons=("Exact independent-unit split and benchmark objectives are not frozen.",),
        reviewed_by=("benchmark-contract-test",),
        reviewed_on=date(2026, 8, 9),
    )
    return data.BenchmarkArtifact(definition=definition, admission=admission), manifest


def test_verifier_re_resolves_exact_scientific_and_legal_assessment(query: StateQuery) -> None:
    benchmark, manifest = blocked_evidence_artifact(population_query(query))
    verification = data.verify_benchmark_artifact(
        benchmark,
        {"claim-evidence": manifest},
    )
    assert verification.evidence_resolutions_verified is True
    assert verification.assessment_and_permission_gates_passed is True
    assert verification.performance_gates_passed is False
    assert verification.admission_ready is False
    assert verification.declared_status is data.BenchmarkAdmissionStatus.BLOCKED

    old_binding = benchmark.definition.evidence_bindings[0]
    old_target = old_binding.scope_binding.target_mappings[0]
    partial_target = revalidate(
        data.EvidenceTargetMapping,
        old_target,
        assessment_modalities=(old_target.assessment_modalities[0],),
    )
    partial_scope = revalidate(
        data.EvidenceScopeBinding,
        old_binding.scope_binding,
        target_mappings=(partial_target,),
    )
    partial_binding = revalidate(
        data.BenchmarkEvidenceBinding,
        old_binding,
        scope_binding=partial_scope,
    )
    partial_definition = revalidate(
        data.BenchmarkDefinition,
        benchmark.definition,
        evidence_bindings=(partial_binding,),
    )
    partial_admission = revalidate(
        data.BenchmarkAdmission,
        benchmark.admission,
        definition_fingerprint=partial_definition.fingerprint,
    )
    partial = data.BenchmarkArtifact(
        definition=partial_definition,
        admission=partial_admission,
    )
    with pytest.raises(ValueError, match="exactly cover assessment-scope modalities"):
        data.verify_benchmark_artifact(partial, {"claim-evidence": manifest})

    resolution = benchmark.admission.evidence_resolutions[0].model_copy(
        update={"resolution_fingerprint": "0" * 64}
    )
    admission = revalidate(
        data.BenchmarkAdmission,
        benchmark.admission,
        evidence_resolutions=(resolution,),
    )
    drifted = data.BenchmarkArtifact(
        definition=benchmark.definition,
        admission=admission,
    )
    with pytest.raises(ValueError, match="resolution has changed"):
        data.verify_benchmark_artifact(drifted, {"claim-evidence": manifest})


def test_verifier_rejects_typed_objective_identity_drift(query: StateQuery) -> None:
    benchmark, manifest = blocked_evidence_artifact(
        population_query(query),
        use_metric_assessment=True,
    )
    old_binding = benchmark.definition.evidence_bindings[0]
    identity = old_binding.assessment_identity
    assert isinstance(identity, data.MetricAssessmentIdentity)
    wrong_identity = data.MetricAssessmentIdentity(
        metric_id=identity.metric_id,
        metric_family=data.MetricFamily.CALIBRATION,
        partition_purpose=identity.partition_purpose,
    )
    binding = revalidate(
        data.BenchmarkEvidenceBinding,
        old_binding,
        assessment_identity=wrong_identity,
    )
    definition = revalidate(
        data.BenchmarkDefinition,
        benchmark.definition,
        evidence_bindings=(binding,),
    )
    admission = revalidate(
        data.BenchmarkAdmission,
        benchmark.admission,
        definition_fingerprint=definition.fingerprint,
    )
    drifted = data.BenchmarkArtifact(definition=definition, admission=admission)
    with pytest.raises(ValueError, match="metric assessment identity does not match"):
        data.verify_benchmark_artifact(drifted, {"metric-evidence": manifest})

    loss_benchmark, loss_manifest = blocked_evidence_artifact(
        population_query(query),
        use_loss_assessment=True,
    )
    old_loss_binding = loss_benchmark.definition.evidence_bindings[0]
    loss_binding = revalidate(
        data.BenchmarkEvidenceBinding,
        old_loss_binding,
        assessment_identity=data.LossAssessmentIdentity(
            loss_kind=LossKind.FUNCTIONAL_OUTCOME,
        ),
    )
    loss_definition = revalidate(
        data.BenchmarkDefinition,
        loss_benchmark.definition,
        evidence_bindings=(loss_binding,),
    )
    loss_admission = revalidate(
        data.BenchmarkAdmission,
        loss_benchmark.admission,
        definition_fingerprint=loss_definition.fingerprint,
    )
    loss_drift = data.BenchmarkArtifact(
        definition=loss_definition,
        admission=loss_admission,
    )
    with pytest.raises(ValueError, match="loss assessment identity does not match"):
        data.verify_benchmark_artifact(loss_drift, {"loss-evidence": loss_manifest})
