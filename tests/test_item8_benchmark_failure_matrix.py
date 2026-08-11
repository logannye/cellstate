from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError
from test_item8_benchmark_contracts import (
    admitted_artifact,
    artifact,
    baseline_reference,
    blocked_evidence_artifact,
    exact_id_membership,
    implementation,
    metric_reference,
    planned_binding,
    population_query,
    rebuild_case_set,
    revalidate,
    unit,
)

import cellstate.data as data
import cellstate.data.benchmarks as contracts
from cellstate.domain import CausalStatus, StateQuery


def _artifact_with_admission(
    benchmark: data.BenchmarkArtifact,
    admission: data.BenchmarkAdmission,
) -> data.BenchmarkArtifact:
    return data.BenchmarkArtifact(
        definition=benchmark.definition,
        leakage_audit=benchmark.leakage_audit,
        admission=admission,
    )


def _admission_with_candidate(
    benchmark: data.BenchmarkArtifact,
    result: data.MetricResult,
) -> data.BenchmarkAdmission:
    return revalidate(
        data.BenchmarkAdmission,
        benchmark.admission,
        metric_results=(result,),
    )


def _paired_comparison(
    benchmark: data.BenchmarkArtifact,
    **updates: object,
) -> data.PairedBaselineComparisonResult:
    case_set = benchmark.definition.evaluation_case_set
    assert case_set is not None
    payload: dict[str, object] = {
        "comparison_id": "candidate-vs-training-mean",
        "metric": metric_reference(benchmark.definition.metrics[0]),
        "partition_id": "p4-test",
        "baseline": baseline_reference(benchmark.definition.baselines[0]),
        "effect_scale": data.BaselineMarginMode.ABSOLUTE_DIFFERENCE,
        "effect_definition": "candidate_minus_baseline_v1",
        "point_effect": -0.1,
        "one_sided_confidence_bound": -0.05,
        "bound_kind": data.PairedConfidenceBoundKind.UPPER,
        "confidence_level": 0.95,
        "evaluated_case_ids": case_set.partition_memberships[-1].evaluation_unit_ids,
        "dependence_ids": ("well",),
        "paired_block_membership_artifact": artifact("paired-block-membership"),
        "result_artifact": artifact("paired-result"),
    }
    payload.update(updates)
    return data.PairedBaselineComparisonResult.model_validate(payload)


@pytest.mark.parametrize(
    ("uri", "message"),
    (
        ("file:///tmp/result.json", "absolute remote URI"),
        ("https:///result.json", "public, absolute, and credential-free"),
        ("https://user:secret@example.org/result.json", "credential-free"),
        ("https://127.0.0.1/result.json", "must not resolve to localhost"),
    ),
)
def test_content_addressed_artifacts_reject_nonpublic_or_mutable_locations(
    uri: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        data.ContentAddressedArtifact(
            artifact_id="unsafe-artifact",
            uri=uri,
            sha256="A" * 64,
            byte_count=1,
            media_type="application/json",
        )


@pytest.mark.parametrize(
    ("canonical_json", "message"),
    (
        ("not-json", "valid JSON"),
        ("NaN", "numeric values must be finite"),
        ('{"b": 2, "a": 1}', "canonical compact JSON"),
    ),
)
def test_query_grid_values_are_finite_canonical_json(
    canonical_json: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        data.QueryParameterValue(
            value_id="dose",
            canonical_json=canonical_json,
        )


def test_query_grid_identity_and_fingerprint_are_canonical(query: StateQuery) -> None:
    value = data.QueryParameterValue(value_id="dose-1", canonical_json="1")
    axis = data.QueryParameterAxis(
        axis_id="dose",
        query_path="intervention_space.0.dose",
        values=(value,),
    )
    grid = data.QueryParameterGrid(
        grid_id="dose-grid",
        grid_version="1.0.0",
        query_fingerprint=query.fingerprint.upper(),
        axes=(axis,),
    )

    assert grid.query_fingerprint == query.fingerprint
    assert len(grid.fingerprint) == 64
    with pytest.raises(ValidationError, match="must be unique"):
        revalidate(data.QueryParameterAxis, axis, values=(value, value))
    with pytest.raises(ValidationError, match="must be unique"):
        revalidate(data.QueryParameterGrid, grid, axes=(axis, axis))


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("query_fingerprint", "0" * 64, "fingerprint must match"),
        ("query_schema_version", "1.0", "Input should be '2.0'"),
    ),
)
def test_state_query_binding_rejects_declared_identity_drift(
    query: StateQuery,
    field: str,
    replacement: object,
    message: str,
) -> None:
    binding = admitted_artifact(query).definition.query
    payload = binding.model_dump(mode="python")
    payload[field] = replacement
    with pytest.raises(ValidationError, match=message):
        data.StateQueryBinding.model_validate(payload)


@pytest.mark.parametrize(
    ("artifact_update", "message"),
    (
        ({"sha256": "0" * 64}, "canonical StateQuery bytes"),
        ({"byte_count": 1}, "byte count"),
        ({"media_type": "text/plain"}, "application/json"),
    ),
)
def test_state_query_binding_rejects_transport_artifact_drift(
    query: StateQuery,
    artifact_update: dict[str, object],
    message: str,
) -> None:
    binding = admitted_artifact(query).definition.query
    query_artifact = revalidate(
        data.ContentAddressedArtifact,
        binding.query_artifact,
        **artifact_update,
    )
    with pytest.raises(ValidationError, match=message):
        revalidate(data.StateQueryBinding, binding, query_artifact=query_artifact)


def test_state_query_binding_rejects_parameter_grid_for_another_query(
    query: StateQuery,
) -> None:
    binding = admitted_artifact(query).definition.query
    grid = data.QueryParameterGrid(
        grid_id="wrong-query-grid",
        grid_version="1.0.0",
        query_fingerprint="0" * 64,
        axes=(
            data.QueryParameterAxis(
                axis_id="dose",
                query_path="intervention_space.0",
                values=(data.QueryParameterValue(value_id="dose-1", canonical_json="1"),),
            ),
        ),
    )
    with pytest.raises(ValidationError, match="grid must bind the exact StateQuery"):
        revalidate(data.StateQueryBinding, binding, parameter_grid=grid)


def test_scope_requires_one_cutoff_and_exact_claim_for_transport(query: StateQuery) -> None:
    scope = admitted_artifact(query).definition.scope
    field_cutoff = revalidate(
        data.BenchmarkScope,
        scope,
        inference_cutoff_seconds=None,
        inference_cutoff_field="assignment_time_seconds",
    )
    assert field_cutoff.inference_cutoff_field == "assignment_time_seconds"

    with pytest.raises(ValidationError, match="exactly one fixed or field cutoff"):
        revalidate(
            data.BenchmarkScope,
            scope,
            inference_cutoff_field="assignment_time_seconds",
        )
    with pytest.raises(ValidationError, match="exactly one fixed or field cutoff"):
        revalidate(data.BenchmarkScope, scope, inference_cutoff_seconds=None)
    with pytest.raises(ValidationError, match="counterfactual-generalization claim"):
        revalidate(
            data.BenchmarkScope,
            scope,
            forecast_causal_status=CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS,
        )


def test_one_physical_binding_cannot_alias_distinct_manifest_versions(
    query: StateQuery,
) -> None:
    benchmark = admitted_artifact(query)
    first, second = benchmark.definition.evidence_bindings
    drifted = revalidate(
        data.BenchmarkEvidenceBinding,
        second,
        dataset_version="different-release",
    )
    with pytest.raises(ValidationError, match="one exact manifest artifact"):
        revalidate(
            data.BenchmarkDefinition,
            benchmark.definition,
            evidence_bindings=(first, drifted),
        )


def test_evidence_binding_rejects_claim_and_manifest_borrowing(query: StateQuery) -> None:
    benchmark = admitted_artifact(query)
    binding = benchmark.definition.evidence_bindings[0]
    with pytest.raises(ValidationError, match="typed assessment identity"):
        revalidate(
            data.BenchmarkEvidenceBinding,
            binding,
            assessment_kind=data.AssessmentKind.METRIC,
        )

    claim_identity = data.ClaimAssessmentIdentity(
        claim=data.ScientificClaim.POPULATION_DYNAMICS,
    )
    with pytest.raises(ValidationError, match="claim identity"):
        revalidate(
            data.BenchmarkEvidenceBinding,
            binding,
            assessment_kind=data.AssessmentKind.CLAIM,
            assessment_identity=claim_identity,
        )

    wrong_reference = binding.assessment_reference.model_copy(
        update={"dataset_manifest_fingerprint": "0" * 64}
    )
    with pytest.raises(ValidationError, match="exact dataset manifest"):
        revalidate(
            data.BenchmarkEvidenceBinding,
            binding,
            assessment_reference=wrong_reference,
        )

    wrong_manifest_artifact = revalidate(
        data.ContentAddressedArtifact,
        binding.manifest_artifact,
        media_type="text/plain",
    )
    with pytest.raises(ValidationError, match="canonical manifest JSON"):
        revalidate(
            data.BenchmarkEvidenceBinding,
            binding,
            manifest_artifact=wrong_manifest_artifact,
        )


def test_target_mapping_requires_an_exact_observed_modality_or_readout(
    query: StateQuery,
) -> None:
    mapping = (
        admitted_artifact(query).definition.evidence_bindings[0].scope_binding.target_mappings[0]
    )
    with pytest.raises(ValidationError, match="requires a modality or functional readout"):
        revalidate(
            data.EvidenceTargetMapping,
            mapping,
            assessment_modalities=(),
            assessment_functional_readout_ids=(),
        )


@pytest.mark.parametrize(
    ("role", "interventions", "controls", "message"),
    (
        (data.EvaluationCaseRole.TREATED, (), ("control",), "exactly one intervention"),
        (data.EvaluationCaseRole.TREATED, ("action",), (), "matched-control unit IDs"),
        (data.EvaluationCaseRole.TREATED, ("action",), ("self",), "own matched control"),
        (
            data.EvaluationCaseRole.MATCHED_NO_ACTION_CONTROL,
            ("action",),
            (),
            "zero actions",
        ),
    ),
)
def test_evaluation_case_roles_fail_closed(
    query: StateQuery,
    role: data.EvaluationCaseRole,
    interventions: tuple[str, ...],
    controls: tuple[str, ...],
    message: str,
) -> None:
    case_set = admitted_artifact(query).definition.evaluation_case_set
    assert case_set is not None
    treated = next(case for case in case_set.cases if case.role is data.EvaluationCaseRole.TREATED)
    evaluation_unit_id = "self" if controls == ("self",) else treated.evaluation_unit_id
    with pytest.raises(ValidationError, match=message):
        revalidate(
            data.BenchmarkEvaluationCase,
            treated,
            role=role,
            evaluation_unit_id=evaluation_unit_id,
            intervention_spec_ids=interventions,
            matched_control_evaluation_unit_ids=controls,
        )


def test_case_set_rejects_duplicate_units_and_metadata_counts(query: StateQuery) -> None:
    case_set = admitted_artifact(query).definition.evaluation_case_set
    assert case_set is not None
    first, second, *rest = case_set.cases
    duplicate = second.model_copy(update={"evaluation_unit_id": first.evaluation_unit_id})
    with pytest.raises(ValidationError, match="unit IDs must be unique"):
        rebuild_case_set(case_set, (first, duplicate, *rest))

    with pytest.raises(ValidationError, match="case count"):
        revalidate(data.BenchmarkEvaluationCaseSet, case_set, case_count=case_set.case_count + 1)
    with pytest.raises(ValidationError, match="intervention multiplicities"):
        declared = case_set.intervention_case_counts[0]
        revalidate(
            data.BenchmarkEvaluationCaseSet,
            case_set,
            intervention_case_counts=(
                declared.model_copy(update={"case_count": declared.case_count + 1}),
            ),
        )
    with pytest.raises(ValidationError, match="no-action control count"):
        revalidate(
            data.BenchmarkEvaluationCaseSet,
            case_set,
            no_action_control_case_count=case_set.no_action_control_case_count + 1,
        )


def test_case_set_rejects_context_partition_and_membership_drift(query: StateQuery) -> None:
    case_set = admitted_artifact(query).definition.evaluation_case_set
    assert case_set is not None
    first, *rest = case_set.cases

    unknown_context = first.model_copy(update={"context_id": "undeclared-context"})
    with pytest.raises(ValidationError, match="exact declared context"):
        rebuild_case_set(case_set, (unknown_context, *rest))

    partition_case = next(case for case in case_set.cases if case.partition_id == "p1-train")
    unknown_partition = partition_case.model_copy(update={"partition_id": "undeclared-partition"})
    with pytest.raises(ValidationError, match="undeclared partition"):
        rebuild_case_set(
            case_set,
            tuple(
                unknown_partition if case.case_id == partition_case.case_id else case
                for case in case_set.cases
            ),
        )

    bad_artifact = revalidate(
        data.ContentAddressedArtifact,
        case_set.case_artifact,
        media_type="text/plain",
    )
    with pytest.raises(ValidationError, match="exact canonical case array"):
        revalidate(data.BenchmarkEvaluationCaseSet, case_set, case_artifact=bad_artifact)

    first_membership, *remaining_memberships = case_set.partition_memberships
    wrong_ids = first_membership.evaluation_unit_ids.model_copy(
        update={"id_count": first_membership.evaluation_unit_ids.id_count + 1}
    )
    with pytest.raises(ValidationError, match="exact evaluation-unit IDs"):
        revalidate(
            data.BenchmarkEvaluationCaseSet,
            case_set,
            partition_memberships=(
                first_membership.model_copy(update={"evaluation_unit_ids": wrong_ids}),
                *remaining_memberships,
            ),
        )


def test_case_set_rejects_unknown_noncontrol_and_mismatched_controls(query: StateQuery) -> None:
    case_set = admitted_artifact(query).definition.evaluation_case_set
    assert case_set is not None
    treated = next(
        case
        for case in case_set.cases
        if case.partition_id == "p1-train" and case.role is data.EvaluationCaseRole.TREATED
    )

    unknown = treated.model_copy(
        update={"matched_control_evaluation_unit_ids": ("unknown-control",)}
    )
    with pytest.raises(ValidationError, match="unknown matched control"):
        rebuild_case_set(
            case_set,
            tuple(unknown if case.case_id == treated.case_id else case for case in case_set.cases),
        )

    control_id = treated.matched_control_evaluation_unit_ids[0]
    control = next(case for case in case_set.cases if case.evaluation_unit_id == control_id)
    mismatched = control.model_copy(
        update={
            "matching_stratum_id": "other-stratum",
            "matching_stratum_fingerprint": "0" * 64,
        }
    )
    with pytest.raises(ValidationError, match="share exact context and stratum"):
        rebuild_case_set(
            case_set,
            tuple(
                mismatched if case.case_id == control.case_id else case for case in case_set.cases
            ),
        )


def test_case_set_rejects_a_matched_control_from_another_horizon(
    query: StateQuery,
) -> None:
    case_set = admitted_artifact(query).definition.evaluation_case_set
    assert case_set is not None
    treated = next(case for case in case_set.cases if case.role is data.EvaluationCaseRole.TREATED)
    control_id = treated.matched_control_evaluation_unit_ids[0]
    control = next(case for case in case_set.cases if case.evaluation_unit_id == control_id)
    wrong_horizon_control = control.model_copy(update={"horizon_name": "later-horizon"})

    with pytest.raises(ValidationError, match="same horizon"):
        rebuild_case_set(
            case_set,
            tuple(
                wrong_horizon_control if case.case_id == control.case_id else case
                for case in case_set.cases
            ),
        )


def test_case_set_rejects_a_treated_case_used_as_control(query: StateQuery) -> None:
    case_set = admitted_artifact(query).definition.evaluation_case_set
    assert case_set is not None
    treated = next(
        case
        for case in case_set.cases
        if case.partition_id == "p1-train" and case.role is data.EvaluationCaseRole.TREATED
    )
    control_id = treated.matched_control_evaluation_unit_ids[0]
    control = next(case for case in case_set.cases if case.evaluation_unit_id == control_id)
    replacement_control = next(
        case
        for case in reversed(case_set.cases)
        if case.partition_id == treated.partition_id
        and case.role is data.EvaluationCaseRole.TREATED
        and case.case_id != treated.case_id
    )
    promoted = control.model_copy(
        update={
            "role": data.EvaluationCaseRole.TREATED,
            "intervention_spec_ids": replacement_control.intervention_spec_ids,
            "matched_control_evaluation_unit_ids": (replacement_control.evaluation_unit_id,),
        }
    )
    demoted = replacement_control.model_copy(
        update={
            "role": data.EvaluationCaseRole.MATCHED_NO_ACTION_CONTROL,
            "intervention_spec_ids": (),
            "matched_control_evaluation_unit_ids": (),
        }
    )
    cases = tuple(
        promoted
        if case.case_id == control.case_id
        else demoted
        if case.case_id == replacement_control.case_id
        else case
        for case in case_set.cases
    )
    with pytest.raises(ValidationError, match="explicit no-action cases"):
        rebuild_case_set(case_set, cases)


def test_protected_split_closure_rejects_ambiguous_unit_hierarchies(
    query: StateQuery,
) -> None:
    plan = admitted_artifact(query).definition.split_plan
    assert plan is not None
    closure = plan.protected_group_closures[0]
    well, cell = closure.unit_ancestry
    sample = unit(data.ExperimentalUnitLevel.SAMPLE, "sample_id")
    well_group = closure.protected_groups[0]
    cell_group = data.ProtectedGroupBinding(
        unit=cell,
        reasons=(data.ProtectedGroupReason.METRIC_EVALUATION,),
    )
    sample_group = data.ProtectedGroupBinding(
        unit=sample,
        reasons=(data.ProtectedGroupReason.DEFAULT_SPLIT,),
    )

    with pytest.raises(ValidationError, match="ancestry members must be unique"):
        revalidate(data.ProtectedGroupClosure, closure, unit_ancestry=(well, well))
    with pytest.raises(ValidationError, match="finest declared ancestry"):
        revalidate(data.ProtectedGroupClosure, closure, record_unit=well)
    with pytest.raises(ValidationError, match="must occur in the unit ancestry"):
        revalidate(data.ProtectedGroupClosure, closure, assignment_unit=sample)
    with pytest.raises(ValidationError, match="protected-group units must be unique"):
        revalidate(
            data.ProtectedGroupClosure,
            closure,
            protected_groups=(well_group, well_group),
        )
    with pytest.raises(ValidationError, match="protected groups must be sorted"):
        revalidate(
            data.ProtectedGroupClosure,
            closure,
            protected_groups=(well_group, cell_group),
        )
    with pytest.raises(ValidationError, match="protected group must occur"):
        revalidate(
            data.ProtectedGroupClosure,
            closure,
            protected_groups=(sample_group, well_group),
        )
    unassigned_group = revalidate(
        data.ProtectedGroupBinding,
        well_group,
        reasons=(
            data.ProtectedGroupReason.METRIC_EVALUATION,
            data.ProtectedGroupReason.OBJECTIVE_REQUIRED_SPLIT,
        ),
    )
    with pytest.raises(ValidationError, match="explicit protected group"):
        revalidate(
            data.ProtectedGroupClosure,
            closure,
            protected_groups=(unassigned_group,),
        )


def test_partition_membership_counts_cannot_exceed_descendant_records(
    query: StateQuery,
) -> None:
    plan = admitted_artifact(query).definition.split_plan
    assert plan is not None
    membership = plan.partitions[0].materialized_membership
    assert membership is not None
    too_many_units = membership.assignment_unit_ids.model_copy(
        update={"id_count": membership.record_ids.id_count + 1}
    )
    with pytest.raises(ValidationError, match="cannot exceed descendant record count"):
        revalidate(
            data.ExplicitPartitionMembership,
            membership,
            assignment_unit_ids=too_many_units,
        )

    universe = plan.universes[0]
    with pytest.raises(ValidationError, match="cannot exceed record count"):
        revalidate(
            data.PartitionUniverse,
            universe,
            assignment_unit_ids=universe.assignment_unit_ids.model_copy(
                update={"id_count": universe.record_ids.id_count + 1}
            ),
        )

    with pytest.raises(ValidationError, match="excluded assignment-unit count"):
        data.ExcludedPartitionMembership(
            assignment_unit_ids=universe.assignment_unit_ids,
            record_ids=universe.record_ids.model_copy(update={"id_count": 1}),
            descendant_closure_artifact=artifact("excluded-descendants"),
            reason_codes_artifact=artifact("excluded-reasons"),
        )


def test_generated_partition_must_materialize_the_declared_assignment_unit(
    query: StateQuery,
) -> None:
    plan = admitted_artifact(query).definition.split_plan
    assert plan is not None
    membership = plan.partitions[0].materialized_membership
    assert membership is not None
    cell = unit(data.ExperimentalUnitLevel.CELL, "cell_id")
    wrong_materialization = revalidate(
        data.ExplicitPartitionMembership,
        membership,
        assignment_unit=cell,
    )
    with pytest.raises(ValidationError, match="declared assignment unit"):
        data.PartitionGenerationSpec(
            generator=implementation("partition-generator"),
            source_universe_fingerprint=plan.universes[0].fingerprint,
            assignment_unit=plan.universes[0].assignment_unit,
            seed=7,
            materialized_membership=wrong_materialization,
        )


@pytest.mark.parametrize(
    ("scenario", "message"),
    (
        ("dataset-coverage", "cover exact physical datasets"),
        ("universe-units", "universe units must match"),
        ("assignment-unit", "assignment unit must match"),
        ("generator-fingerprint", "bind the exact source universe"),
        ("record-unit", "record unit must match"),
        ("protected-membership", "membership for every protected group"),
        ("record-exhaustion", "records plus exclusions must exhaust"),
        ("unit-exhaustion", "units plus exclusions must exhaust"),
    ),
)
def test_split_plan_materialization_is_physically_complete(
    query: StateQuery,
    scenario: str,
    message: str,
) -> None:
    plan = admitted_artifact(query).definition.split_plan
    assert plan is not None
    partitions = list(plan.partitions)
    universes = plan.universes
    first = partitions[0]
    membership = first.materialized_membership
    assert membership is not None
    cell = unit(data.ExperimentalUnitLevel.CELL, "cell_id")

    if scenario == "dataset-coverage":
        partitions[0] = first.model_copy(
            update={"physical_dataset_binding_id": "unknown-physical-dataset"}
        )
    elif scenario == "universe-units":
        universes = (revalidate(data.PartitionUniverse, universes[0], assignment_unit=cell),)
    elif scenario == "assignment-unit":
        partitions[0] = first.model_copy(
            update={
                "membership": revalidate(
                    data.ExplicitPartitionMembership,
                    membership,
                    assignment_unit=cell,
                )
            }
        )
    elif scenario == "generator-fingerprint":
        partitions[0] = first.model_copy(
            update={
                "membership": data.PartitionGenerationSpec(
                    generator=implementation("wrong-universe-generator"),
                    source_universe_fingerprint="0" * 64,
                    assignment_unit=membership.assignment_unit,
                    seed=11,
                )
            }
        )
    elif scenario == "record-unit":
        partitions[0] = first.model_copy(
            update={
                "membership": revalidate(
                    data.ExplicitPartitionMembership,
                    membership,
                    record_unit=membership.assignment_unit,
                )
            }
        )
    elif scenario == "protected-membership":
        partitions[0] = first.model_copy(
            update={
                "membership": revalidate(
                    data.ExplicitPartitionMembership,
                    membership,
                    protected_group_memberships=(
                        data.ProtectedGroupMembership(
                            unit=cell,
                            membership=membership.record_ids,
                        ),
                    ),
                )
            }
        )
    elif scenario == "record-exhaustion":
        partitions[0] = first.model_copy(
            update={
                "membership": revalidate(
                    data.ExplicitPartitionMembership,
                    membership,
                    record_ids=membership.record_ids.model_copy(
                        update={"id_count": membership.record_ids.id_count - 1}
                    ),
                )
            }
        )
    else:
        partitions[0] = first.model_copy(
            update={
                "membership": revalidate(
                    data.ExplicitPartitionMembership,
                    membership,
                    assignment_unit_ids=membership.assignment_unit_ids.model_copy(
                        update={"id_count": membership.assignment_unit_ids.id_count - 1}
                    ),
                )
            }
        )

    with pytest.raises(ValidationError, match=message):
        revalidate(
            data.BenchmarkSplitPlan,
            plan,
            universes=universes,
            partitions=tuple(partitions),
        )


@pytest.mark.parametrize(
    ("scheme", "group", "weights", "message"),
    (
        (
            data.MetricWeightingScheme.EQUAL_GROUP_THEN_EQUAL_EVALUATION_UNIT,
            None,
            None,
            "exactly one declared group",
        ),
        (
            data.MetricWeightingScheme.CONTENT_ADDRESSED_FIXED,
            "compound",
            None,
            "content-addressed weight table",
        ),
        (
            data.MetricWeightingScheme.EQUAL_EVALUATION_UNIT,
            "compound",
            None,
            "un-grouped weighting",
        ),
    ),
)
def test_metric_weighting_never_infers_group_or_record_weights(
    scheme: data.MetricWeightingScheme,
    group: str | None,
    weights: data.ContentAddressedArtifact | None,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        data.MetricWeightingPolicy(
            scheme=scheme,
            group_dependence_id=group,
            fixed_weights_artifact=weights,
        )


def test_metric_dependence_requires_typed_unit_identity(query: StateQuery) -> None:
    metric = admitted_artifact(query).definition.metrics[0]
    dependence = metric.uncertainty.dependence_units[0]
    with pytest.raises(ValidationError, match="require, and only they permit"):
        revalidate(
            data.MetricDependenceUnit,
            dependence,
            kind=data.MetricDependenceKind.INTERVENTION_CONDITION,
        )
    with pytest.raises(ValidationError, match="require, and only they permit"):
        revalidate(data.MetricDependenceUnit, dependence, experimental_unit=None)

    sample = unit(data.ExperimentalUnitLevel.SAMPLE, "sample_id")
    with pytest.raises(ValidationError, match="identity must match"):
        revalidate(data.MetricDependenceUnit, dependence, experimental_unit=sample)


def test_metric_uncertainty_rejects_unknown_cyclic_and_inadequate_blocks(
    query: StateQuery,
) -> None:
    uncertainty = admitted_artifact(query).definition.metrics[0].uncertainty
    well = uncertainty.dependence_units[0]
    unknown_parent = well.model_copy(update={"parent_dependence_ids": ("unknown",)})
    with pytest.raises(ValidationError, match="known and non-reflexive"):
        revalidate(data.MetricUncertaintySpec, uncertainty, dependence_units=(unknown_parent,))

    compound = revalidate(
        data.MetricDependenceUnit,
        well,
        dependence_id="compound",
        kind=data.MetricDependenceKind.INTERVENTION_CONDITION,
        experimental_unit=None,
        parent_dependence_ids=("well",),
    )
    cyclic_well = well.model_copy(update={"parent_dependence_ids": ("compound",)})
    with pytest.raises(ValidationError, match="hierarchy must be acyclic"):
        revalidate(
            data.MetricUncertaintySpec,
            uncertainty,
            resampling_scheme=data.MetricResamplingScheme.CLUSTERED,
            dependence_units=(compound, cyclic_well),
        )

    with pytest.raises(ValidationError, match="iid uncertainty requires exactly one"):
        revalidate(
            data.MetricUncertaintySpec,
            uncertainty,
            dependence_units=(compound, well),
        )
    with pytest.raises(ValidationError, match="requires at least two groupings"):
        revalidate(
            data.MetricUncertaintySpec,
            uncertainty,
            resampling_scheme=data.MetricResamplingScheme.MULTIWAY_CLUSTERED,
        )


def test_metric_definition_rejects_missingness_and_replication_shortcuts(
    query: StateQuery,
) -> None:
    metric = admitted_artifact(query).definition.metrics[0]
    with pytest.raises(ValidationError, match="masked metrics require"):
        revalidate(
            data.BenchmarkMetricDefinition,
            metric,
            missingness_policy=data.MetricMissingnessPolicy.MASK_WITH_REPORTED_DENOMINATOR,
        )
    with pytest.raises(ValidationError, match="only masked metrics permit"):
        revalidate(data.BenchmarkMetricDefinition, metric, minimum_coverage=0.9)

    sample = unit(data.ExperimentalUnitLevel.SAMPLE, "sample_id")
    with pytest.raises(ValidationError, match="resample the declared evaluation unit"):
        revalidate(data.BenchmarkMetricDefinition, metric, evaluation_unit=sample)

    grouped_weighting = data.MetricWeightingPolicy(
        scheme=data.MetricWeightingScheme.EQUAL_GROUP_THEN_EQUAL_EVALUATION_UNIT,
        group_dependence_id="compound",
    )
    with pytest.raises(ValidationError, match="group must resolve"):
        revalidate(data.BenchmarkMetricDefinition, metric, weighting=grouped_weighting)

    with pytest.raises(ValidationError, match="pooled-record aggregation"):
        revalidate(
            data.BenchmarkMetricDefinition,
            metric,
            aggregation=data.MetricAggregation.POOLED_RECORD,
        )
    with pytest.raises(ValidationError, match="pooled-record aggregation"):
        revalidate(
            data.BenchmarkMetricDefinition,
            metric,
            weighting=data.MetricWeightingPolicy(
                scheme=data.MetricWeightingScheme.RECORD_COUNT_WEIGHTED,
            ),
        )


def test_specification_only_implementations_require_canonical_blockers() -> None:
    valid = planned_binding("planned-score")
    assert valid.kind == "specification_only"
    with pytest.raises(ValidationError, match="must be sorted"):
        revalidate(
            data.SpecificationOnlyImplementationBinding,
            valid,
            blockers=("z blocker", "a blocker"),
        )


@pytest.mark.parametrize(
    ("update", "message"),
    (
        ({"seeds": (1, 0)}, "unique and sorted"),
        (
            {"training_partition_ids": (), "fixed_model_artifact": None},
            "requires training partitions or a fixed model artifact",
        ),
    ),
)
def test_baseline_definition_is_reproducible(
    query: StateQuery,
    update: dict[str, object],
    message: str,
) -> None:
    baseline = admitted_artifact(query).definition.baselines[0]
    with pytest.raises(ValidationError, match=message):
        revalidate(data.BenchmarkBaselineDefinition, baseline, **update)


@pytest.mark.parametrize(
    ("update", "message"),
    (
        (
            {"comparisons_expected": None},
            "require complete comparison and overlap counts",
        ),
        (
            {"comparisons_completed": 0},
            "complete every declared comparison",
        ),
        ({"overlap_count": 1}, "passed overlap checks require zero overlap"),
    ),
)
def test_overlap_audit_checks_are_complete_and_zero_overlap(
    query: StateQuery,
    update: dict[str, object],
    message: str,
) -> None:
    benchmark = admitted_artifact(query)
    assert benchmark.leakage_audit is not None
    check = benchmark.leakage_audit.checks[0]
    with pytest.raises(ValidationError, match=message):
        revalidate(data.LeakageAuditCheck, check, **update)


def test_leakage_check_kind_controls_scope_and_failure_evidence(query: StateQuery) -> None:
    benchmark = admitted_artifact(query)
    assert benchmark.leakage_audit is not None
    record_check = benchmark.leakage_audit.checks[0]
    nonoverlap = benchmark.leakage_audit.checks[3]

    with pytest.raises(ValidationError, match="non-overlap checks must omit"):
        revalidate(
            data.LeakageAuditCheck,
            nonoverlap,
            comparisons_expected=1,
            comparisons_completed=1,
            overlap_count=0,
        )
    with pytest.raises(ValidationError, match="physical dataset and unit identity"):
        revalidate(
            data.LeakageAuditCheck,
            record_check,
            kind=data.LeakageCheckKind.PROTECTED_GROUP_DISJOINT,
            protected_unit=None,
        )
    with pytest.raises(ValidationError, match="only protected-group checks"):
        revalidate(
            data.LeakageAuditCheck,
            nonoverlap,
            protected_unit=unit(data.ExperimentalUnitLevel.WELL, "well_id"),
        )
    with pytest.raises(ValidationError, match="require a physical dataset binding"):
        revalidate(
            data.LeakageAuditCheck,
            record_check,
            physical_dataset_binding_id=None,
        )
    with pytest.raises(ValidationError, match="require notes and no blockers"):
        revalidate(data.LeakageAuditCheck, nonoverlap, notes=())
    with pytest.raises(ValidationError, match="require blockers"):
        revalidate(
            data.LeakageAuditCheck,
            nonoverlap,
            status=data.AuditCheckStatus.NOT_ASSESSED,
            notes=(),
        )


@pytest.mark.parametrize(
    ("update", "message"),
    (
        ({"absolute_threshold": None}, "exactly one absolute or baseline threshold"),
        (
            {
                "absolute_threshold": None,
                "baseline_margin": 0.1,
                "baseline_comparator": None,
                "baseline_margin_mode": None,
                "baseline_requirement": None,
            },
            "exact baseline, mode, and requirement",
        ),
        (
            {
                "absolute_threshold": None,
                "baseline_margin": -0.1,
                "baseline_comparator": "baseline",
                "baseline_margin_mode": data.BaselineMarginMode.ABSOLUTE_DIFFERENCE,
                "baseline_requirement": data.BaselineRequirement.NONINFERIOR,
            },
            "nonnegative magnitudes",
        ),
    ),
)
def test_acceptance_thresholds_cannot_leave_semantics_implicit(
    query: StateQuery,
    update: dict[str, object],
    message: str,
) -> None:
    benchmark = admitted_artifact(query)
    rule = benchmark.definition.acceptance_rules[0]
    if update.get("baseline_comparator") == "baseline":
        update["baseline_comparator"] = data.ExactBaselineComparator(
            baseline=baseline_reference(benchmark.definition.baselines[0])
        )
    with pytest.raises(ValidationError, match=message):
        revalidate(data.BenchmarkAcceptanceRule, rule, **update)


def test_acceptance_thresholds_type_relative_margin_and_confidence(query: StateQuery) -> None:
    benchmark = admitted_artifact(query)
    rule = benchmark.definition.acceptance_rules[0]
    comparator = data.ExactBaselineComparator(
        baseline=baseline_reference(benchmark.definition.baselines[0])
    )
    relative = {
        "absolute_threshold": None,
        "baseline_margin": 1.1,
        "baseline_comparator": comparator,
        "baseline_margin_mode": data.BaselineMarginMode.RELATIVE_FRACTION,
        "baseline_requirement": data.BaselineRequirement.NONINFERIOR,
        "estimate": data.ThresholdEstimate.UPPER_CONFIDENCE_BOUND,
        "confidence_level": 0.95,
    }
    with pytest.raises(ValidationError, match=r"lie in \[0, 1\]"):
        revalidate(data.BenchmarkAcceptanceRule, rule, **relative)

    relative["baseline_margin"] = 0.1
    relative["estimate"] = data.ThresholdEstimate.POINT
    relative["confidence_level"] = None
    with pytest.raises(ValidationError, match="one-sided confidence bound"):
        revalidate(data.BenchmarkAcceptanceRule, rule, **relative)

    with pytest.raises(ValidationError, match=r"point rules omit.*confidence"):
        revalidate(data.BenchmarkAcceptanceRule, rule, confidence_level=0.95)


def test_acceptance_group_and_policy_must_form_one_exact_tree() -> None:
    with pytest.raises(ValidationError, match="at least one rule or child group"):
        data.BenchmarkAcceptanceGroup(
            group_id="empty",
            operator=data.AcceptanceGroupOperator.ALL,
        )
    with pytest.raises(ValidationError, match="cannot contain themselves"):
        data.BenchmarkAcceptanceGroup(
            group_id="self",
            operator=data.AcceptanceGroupOperator.ALL,
            child_group_ids=("self",),
        )

    leaf = data.BenchmarkAcceptanceGroup(
        group_id="leaf",
        operator=data.AcceptanceGroupOperator.ALL,
        rule_ids=("rule",),
    )
    with pytest.raises(ValidationError, match="root group must exist"):
        data.BenchmarkAcceptancePolicy(
            policy_id="missing-root",
            policy_version="1",
            root_group_id="root",
            groups=(leaf,),
        )
    root_unknown = data.BenchmarkAcceptanceGroup(
        group_id="root",
        operator=data.AcceptanceGroupOperator.ALL,
        child_group_ids=("unknown",),
    )
    with pytest.raises(ValidationError, match="unknown child group"):
        data.BenchmarkAcceptancePolicy(
            policy_id="unknown-child",
            policy_version="1",
            root_group_id="root",
            groups=(root_unknown,),
        )

    root = data.BenchmarkAcceptanceGroup(
        group_id="root",
        operator=data.AcceptanceGroupOperator.ALL,
        child_group_ids=("left", "right"),
    )
    left = data.BenchmarkAcceptanceGroup(
        group_id="left",
        operator=data.AcceptanceGroupOperator.ALL,
        rule_ids=("same-rule",),
    )
    right = data.BenchmarkAcceptanceGroup(
        group_id="right",
        operator=data.AcceptanceGroupOperator.ANY,
        rule_ids=("same-rule",),
    )
    with pytest.raises(ValidationError, match="exactly one group"):
        data.BenchmarkAcceptancePolicy(
            policy_id="duplicate-rule",
            policy_version="1",
            root_group_id="root",
            groups=(left, right, root),
        )


def test_metric_result_interval_is_complete_and_contains_estimate(query: StateQuery) -> None:
    result = admitted_artifact(query).admission.metric_results[0]
    with pytest.raises(ValidationError, match="supplied together"):
        revalidate(data.MetricResult, result, lower_confidence_bound=None)
    with pytest.raises(ValidationError, match="inside its confidence interval"):
        revalidate(
            data.MetricResult,
            result,
            lower_confidence_bound=0.5,
            upper_confidence_bound=0.6,
        )


@pytest.mark.parametrize(
    ("update", "message"),
    (
        (
            {"effect_definition": "candidate_minus_baseline_over_abs_baseline_v1"},
            "definition must match",
        ),
        (
            {
                "bound_kind": data.PairedConfidenceBoundKind.LOWER,
                "one_sided_confidence_bound": 0.0,
            },
            "lower paired bound cannot exceed",
        ),
        (
            {"one_sided_confidence_bound": -0.2},
            "upper paired bound cannot be below",
        ),
    ),
)
def test_paired_comparison_bound_and_effect_scale_are_coherent(
    query: StateQuery,
    update: dict[str, object],
    message: str,
) -> None:
    benchmark = admitted_artifact(query)
    with pytest.raises(ValidationError, match=message):
        _paired_comparison(benchmark, **update)


@pytest.mark.parametrize(
    ("status", "update", "message"),
    (
        (
            data.BaselineRunStatus.PASSED,
            {"prediction_artifact": None},
            "passed baselines require predictions",
        ),
        (
            data.BaselineRunStatus.NOT_APPLICABLE,
            {"prediction_artifact": None, "metric_results": (), "notes": ()},
            "require a derived explanation",
        ),
        (
            data.BaselineRunStatus.CRASHED,
            {"prediction_artifact": None, "metric_results": (), "blockers": ()},
            "require blockers",
        ),
    ),
)
def test_baseline_runs_fail_closed_for_every_execution_state(
    query: StateQuery,
    status: data.BaselineRunStatus,
    update: dict[str, object],
    message: str,
) -> None:
    run = admitted_artifact(query).admission.baseline_runs[0]
    with pytest.raises(ValidationError, match=message):
        revalidate(data.BaselineRun, run, status=status, **update)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda benchmark, result: result.model_copy(
                update={"metric": result.metric.model_copy(update={"metric_fingerprint": "0" * 64})}
            ),
            "bind exact metric definitions",
        ),
        (
            lambda benchmark, result: result.model_copy(
                update={"partition_id": "unknown-partition"}
            ),
            "unknown partition",
        ),
        (
            lambda benchmark, result: result.model_copy(update={"partition_id": "p3-validation"}),
            "unfrozen evaluation partition",
        ),
        (
            lambda benchmark, result: result.model_copy(
                update={"evaluated_case_ids": exact_id_membership("wrong-cases", ("wrong",))}
            ),
            "exact authoritative case membership",
        ),
        (
            lambda benchmark, result: result.model_copy(
                update={"evaluated_evaluation_units": result.evaluated_evaluation_units + 1}
            ),
            "evaluation-unit count must be exact",
        ),
        (
            lambda benchmark, result: result.model_copy(
                update={"lower_confidence_bound": None, "upper_confidence_bound": None}
            ),
            "require both uncertainty bounds",
        ),
    ),
)
def test_metric_reporting_binds_exact_definition_partition_and_cases(
    query: StateQuery,
    mutate: Callable[[data.BenchmarkArtifact, data.MetricResult], data.MetricResult],
    message: str,
) -> None:
    benchmark = admitted_artifact(query)
    result = mutate(benchmark, benchmark.admission.metric_results[0])
    admission = _admission_with_candidate(benchmark, result)
    with pytest.raises(ValidationError, match=message):
        _artifact_with_admission(benchmark, admission)


def test_admission_reports_every_baseline_and_derived_applicability(query: StateQuery) -> None:
    benchmark = admitted_artifact(query)
    no_runs = revalidate(
        data.BenchmarkAdmission,
        benchmark.admission,
        baseline_runs=(),
    )
    with pytest.raises(ValidationError, match="report every declared baseline"):
        _artifact_with_admission(benchmark, no_runs)

    run = benchmark.admission.baseline_runs[0]
    wrong_fingerprint = revalidate(
        data.BaselineRun,
        run,
        applicability_rule_fingerprint="0" * 64,
    )
    wrong = revalidate(
        data.BenchmarkAdmission,
        benchmark.admission,
        baseline_runs=(wrong_fingerprint,),
    )
    with pytest.raises(ValidationError, match="exact definition and applicability"):
        _artifact_with_admission(benchmark, wrong)

    not_applicable = revalidate(
        data.BaselineRun,
        run,
        status=data.BaselineRunStatus.NOT_APPLICABLE,
        prediction_artifact=None,
        metric_results=(),
        notes=("Derived from the frozen query.",),
    )
    false_na = revalidate(
        data.BenchmarkAdmission,
        benchmark.admission,
        baseline_runs=(not_applicable,),
    )
    with pytest.raises(ValidationError, match="N/A status must be derived"):
        _artifact_with_admission(benchmark, false_na)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda benchmark, comparison: comparison.model_copy(
                update={"metric": comparison.metric.model_copy(update={"metric_id": "unknown"})}
            ),
            "exact metric definition",
        ),
        (
            lambda benchmark, comparison: comparison.model_copy(
                update={
                    "baseline": comparison.baseline.model_copy(update={"baseline_id": "unknown"})
                }
            ),
            "exact baseline definition",
        ),
        (
            lambda benchmark, comparison: comparison.model_copy(
                update={"evaluated_case_ids": exact_id_membership("wrong-pair", ("wrong",))}
            ),
            "exact authoritative case membership",
        ),
        (
            lambda benchmark, comparison: comparison.model_copy(
                update={"dependence_ids": ("unknown",)}
            ),
            "every exact metric dependence block",
        ),
        (
            lambda benchmark, comparison: comparison.model_copy(
                update={"point_effect": 0.0, "one_sided_confidence_bound": 0.05}
            ),
            "point effect must equal",
        ),
    ),
)
def test_paired_results_bind_exact_metric_baseline_cases_and_blocks(
    query: StateQuery,
    mutate: Callable[
        [data.BenchmarkArtifact, data.PairedBaselineComparisonResult],
        data.PairedBaselineComparisonResult,
    ],
    message: str,
) -> None:
    benchmark = admitted_artifact(query)
    comparison = mutate(benchmark, _paired_comparison(benchmark))
    admission = revalidate(
        data.BenchmarkAdmission,
        benchmark.admission,
        paired_baseline_comparisons=(comparison,),
    )
    with pytest.raises(ValidationError, match=message):
        _artifact_with_admission(benchmark, admission)


@pytest.mark.parametrize(
    ("comparison", "value", "threshold", "expected"),
    (
        (data.ThresholdComparison.LESS_THAN, 0.4, 0.5, True),
        (data.ThresholdComparison.LESS_THAN_OR_EQUAL, 0.5, 0.5, True),
        (data.ThresholdComparison.GREATER_THAN, 0.6, 0.5, True),
        (data.ThresholdComparison.GREATER_THAN_OR_EQUAL, 0.5, 0.5, True),
    ),
)
def test_all_typed_threshold_comparisons_have_exact_boundary_semantics(
    comparison: data.ThresholdComparison,
    value: float,
    threshold: float,
    expected: bool,
) -> None:
    assert contracts._threshold_comparison_passes(value, comparison, threshold) is expected


def test_acceptance_estimates_never_substitute_point_for_missing_uncertainty(
    query: StateQuery,
) -> None:
    result = revalidate(
        data.MetricResult,
        admitted_artifact(query).admission.metric_results[0],
        lower_confidence_bound=None,
        upper_confidence_bound=None,
    )
    with pytest.raises(ValueError, match="lower confidence bound"):
        contracts._metric_result_estimate(result, data.ThresholdEstimate.LOWER_CONFIDENCE_BOUND)
    with pytest.raises(ValueError, match="upper confidence bound"):
        contracts._metric_result_estimate(result, data.ThresholdEstimate.UPPER_CONFIDENCE_BOUND)


def test_planned_component_is_verified_but_not_performance_ready(query: StateQuery) -> None:
    benchmark, manifest = blocked_evidence_artifact(population_query(query))
    verification = data.verify_benchmark_artifact(
        benchmark,
        {"claim-evidence": manifest},
    )

    assert verification.verified is True
    assert verification.assessment_and_permission_gates_passed is True
    assert verification.performance_gates_passed is False
    assert verification.admission_ready is False
    assert verification.technical_fixture_eligible is False
    assert verification.blockers == ("benchmark performance gates are incomplete or did not pass",)

    with pytest.raises(ValueError, match="every and only bound dataset manifests"):
        data.verify_benchmark_artifact(benchmark, {})
