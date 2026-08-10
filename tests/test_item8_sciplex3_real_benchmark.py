"""Adversarial checks for the frozen public-real sci-Plex3 K562 component."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from cellstate.data.benchmarks import (
    AcceptanceGroupOperator,
    AuditCheckStatus,
    BaselineMarginMode,
    BaselineRequirement,
    BaselineRunStatus,
    BenchmarkAdmissionStatus,
    BenchmarkArtifact,
    BenchmarkIntent,
    BenchmarkLifecycle,
    BenchmarkPartitionRole,
    BestApplicableBaselineComparator,
    EvaluationCaseRole,
    ExactBaselineComparator,
    SpecificationOnlyImplementationBinding,
    ThresholdEstimate,
    verify_benchmark_artifact,
)
from cellstate.data.manifests import (
    ControlPredicateValueType,
    DatasetManifest,
    DataUseCase,
    ExperimentalUnitLevel,
    PermissionStatus,
    UnitIdentityExpressionKind,
)
from cellstate.domain.common import CausalStatus, canonical_json_bytes
from cellstate.domain.events import ReversibilityStatus
from cellstate.domain.query import AssayPurpose, StateQuery

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks/vertical-a/sciplex3-k562-24h-v1"
BENCHMARK_PATH = BENCHMARK_DIR / "benchmark-artifact.json"
MANIFEST_PATH = ROOT / "data_manifests/reviewed/sciplex3-k562-24h.json"
QUERY_PATH = BENCHMARK_DIR / "state-query.json"
CASE_PATH = BENCHMARK_DIR / "support/evaluation-cases.json"
SCORING_TRANSFORM_PATH = BENCHMARK_DIR / "support/scoring-transform.json"
TARGET_VALUE_SCHEMA_PATH = BENCHMARK_DIR / "support/target-value-schema.json"
SOURCE_ID = "scperturb-v1.4-sciplex3-h5ad"


@pytest.fixture(scope="module")
def manifest() -> DatasetManifest:
    return DatasetManifest.model_validate_json(MANIFEST_PATH.read_bytes())


@pytest.fixture(scope="module")
def query() -> StateQuery:
    return StateQuery.model_validate_json(QUERY_PATH.read_bytes())


@pytest.fixture(scope="module")
def benchmark() -> BenchmarkArtifact:
    return BenchmarkArtifact.model_validate_json(BENCHMARK_PATH.read_bytes())


def _assert_local_artifact_bytes(artifact: object) -> bytes:
    uri = artifact.uri
    relative_path = uri.split("/main/", maxsplit=1)[1]
    payload = (ROOT / relative_path).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == artifact.sha256
    assert len(payload) == artifact.byte_count
    return payload


def test_canonical_files_and_external_verification_pass_at_component_boundary(
    manifest: DatasetManifest,
    query: StateQuery,
    benchmark: BenchmarkArtifact,
) -> None:
    assert MANIFEST_PATH.read_bytes() == manifest.canonical_json_bytes
    assert hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest() == manifest.fingerprint
    assert QUERY_PATH.read_bytes() == canonical_json_bytes(query.model_dump(mode="json"))
    assert hashlib.sha256(QUERY_PATH.read_bytes()).hexdigest() == query.fingerprint
    assert BENCHMARK_PATH.read_bytes() == canonical_json_bytes(benchmark.model_dump(mode="json"))

    case_set = benchmark.definition.evaluation_case_set
    assert case_set is not None
    assert CASE_PATH.read_bytes() == canonical_json_bytes(
        [case.model_dump(mode="json") for case in case_set.cases]
    )
    assert _assert_local_artifact_bytes(case_set.case_artifact) == CASE_PATH.read_bytes()
    for binding in benchmark.definition.evidence_bindings:
        assert _assert_local_artifact_bytes(binding.manifest_artifact) == (
            manifest.canonical_json_bytes
        )

    verification = verify_benchmark_artifact(
        benchmark,
        {binding.binding_id: manifest for binding in benchmark.definition.evidence_bindings},
    )
    assert verification.evidence_resolutions_verified
    assert verification.assessment_and_permission_gates_passed
    assert not verification.performance_gates_passed
    assert not verification.admission_ready
    assert verification.verified
    assert verification.blockers == ("benchmark performance gates are incomplete or did not pass",)


def test_query_is_context_only_and_action_domain_is_exact(
    query: StateQuery,
    benchmark: BenchmarkArtifact,
) -> None:
    assert query.evidence_policy.minimum_observed_measurements == 0
    assert tuple(term.key for term in query.evidence_policy.allowed_modalities) == (
        "cellstate:sciplex3-k562-design-context",
    )
    assert all(
        output.term.key not in {term.key for term in query.evidence_policy.allowed_modalities}
        for output in query.target_outputs
    )
    assert len(query.prediction_horizons) == 1
    assert query.prediction_horizons[0].duration_seconds == 86_400.0
    assert "assigned to one source well" in query.subject.membership_semantics
    assert "recovered" not in query.subject.membership_semantics.casefold()

    target = query.target_outputs[0]
    panel_path = ROOT / "benchmarks/artifacts/sciplex3-k562-24h-v1/feature-panel.json"
    panel_sha256 = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    value_schema_sha256 = hashlib.sha256(TARGET_VALUE_SCHEMA_PATH.read_bytes()).hexdigest()
    assert target.value_schema_reference.fingerprint == value_schema_sha256
    assert target.value_schema_reference.fingerprint != panel_sha256
    assert target.value_schema_reference.reference_id != (
        target.endpoint.protocol_reference.reference_id  # type: ignore[union-attr]
    )
    assert "2000_feature_panel" in target.units
    assert all(
        metric.target_output_keys == (target.term.key,) for metric in benchmark.definition.metrics
    )

    assert len(query.intervention_space) == 752
    assert len({spec.kind.key for spec in query.intervention_space}) == 188
    assert {spec.target for spec in query.intervention_space} == {None}
    assert {
        (spec.dose_domain.minimum, spec.dose_domain.maximum, spec.dose_domain.units)
        for spec in query.intervention_space
    } == {
        (10.0, 10.0, "nM"),
        (100.0, 100.0, "nM"),
        (1_000.0, 1_000.0, "nM"),
        (10_000.0, 10_000.0, "nM"),
    }
    assert all(
        spec.duration_seconds.minimum == spec.duration_seconds.maximum == 86_400.0
        and spec.allowed_reversibility_statuses == (ReversibilityStatus.UNKNOWN,)
        for spec in query.intervention_space
    )

    assert len(query.available_assays) == 1
    endpoint_assay = query.available_assays[0]
    assert endpoint_assay.modality.key == "efo:0009809"
    assert endpoint_assay.purposes == (AssayPurpose.TARGET_ENDPOINT,)
    assert endpoint_assay.cost is None
    assert endpoint_assay.cost_units is None
    assert endpoint_assay.turnaround_seconds is None
    assert query.constraints.maximum_total_assay_cost is None
    assert query.constraints.assay_cost_units is None
    assert query.constraints.maximum_assay_delay_seconds is None


def test_scoring_uses_only_declared_panel_and_zero_total_fails_closed(
    query: StateQuery,
    benchmark: BenchmarkArtifact,
) -> None:
    panel_path = ROOT / "benchmarks/artifacts/sciplex3-k562-24h-v1/feature-panel.json"
    panel = json.loads(panel_path.read_bytes())
    panel_sha256 = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    scoring_bytes = SCORING_TRANSFORM_PATH.read_bytes()
    scoring_sha256 = hashlib.sha256(scoring_bytes).hexdigest()
    scoring = json.loads(scoring_bytes)
    value_schema = json.loads(TARGET_VALUE_SCHEMA_PATH.read_bytes())

    assert value_schema["feature_axis"]["panel_artifact"]["sha256"] == panel_sha256
    assert value_schema["feature_axis"]["feature_count"] == 2_000
    assert (
        value_schema["feature_axis"]["ordered_feature_keys_sha256"]
        == (panel["ordered_feature_keys_sha256"])
    )
    assert value_schema["scoring_transform_artifact"]["sha256"] == scoring_sha256
    assert value_schema["scoring_input_sufficiency"] == {
        "external_inputs_required": [],
        "full_source_axis_counts_required": False,
        "statement": (
            "The declared raw 2,000-coordinate target contains every value needed to compute "
            "the scoring denominator."
        ),
    }

    assert scoring["input_contract"]["feature_count"] == 2_000
    assert not scoring["input_contract"]["full_source_axis_input_allowed"]
    assert scoring["input_contract"]["external_denominator_inputs"] == []
    assert scoring["scoring_transform"] == {
        "denominator_definition": (
            "panel_total = sum(count_i for i in the exact ordered 2,000-feature panel)"
        ),
        "denominator_scope": "declared_ordered_2000_feature_panel_only",
        "formula": "log1p(10000 * count_i / panel_total)",
        "log_base": "e",
        "scale": 10_000,
        "transformation_id": "panel-only-natural-log-cp10k",
        "transformation_version": "1.0.0",
    }
    assert (
        scoring["train_feature_selection_transform"]["source_metadata"] == (panel["transformation"])
    )
    assert (
        scoring["train_feature_selection_transform"]["purpose"]
        == "TRAIN-only feature ranking and selection evidence"
    )
    assert not scoring["train_feature_selection_transform"]["used_for_benchmark_scoring"]
    assert scoring["train_feature_selection_transform"]["full_source_axis_denominator_used"]

    fatal_conditions = set(scoring["validation_policy"]["fatal_conditions"])
    assert fatal_conditions == {
        "coordinate_order_missing_or_not_exactly_bound",
        "nonfinite_coordinate",
        "noninteger_coordinate",
        "negative_coordinate",
        "panel_total_less_than_or_equal_to_zero",
        "vector_length_not_exactly_2000",
    }
    assert scoring["validation_policy"]["zero_panel_total_policy"] == (
        "error_fail_evaluation_no_exclusion_or_imputation"
    )
    assert "do not drop, exclude, impute" in scoring["validation_policy"]["failure_action"]

    metric_suite = json.loads((BENCHMARK_DIR / "support/metric-suite-spec.json").read_bytes())
    assert not metric_suite["feature_panel"]["full_axis_logcp10k_used_for_scoring"]
    assert metric_suite["feature_panel"]["full_axis_logcp10k_purpose"] == (
        "TRAIN-only feature ranking and selection"
    )
    assert metric_suite["scoring_transform"]["sha256"] == scoring_sha256
    assert "panel_total_2000" in metric_suite["metrics"][0]["formula"]
    assert "library_size" not in metric_suite["scoring_transform_rule"]
    baseline_suite = json.loads((BENCHMARK_DIR / "support/baseline-suite-spec.json").read_bytes())
    assert baseline_suite["scoring_contract"]["scoring_transform_artifact"]["sha256"] == (
        scoring_sha256
    )
    assert (
        "identical panel-only transform"
        in baseline_suite["scoring_contract"]["candidate_and_baseline_symmetry"]
    )

    target = query.target_outputs[0]
    assert (
        target.value_schema_reference.fingerprint
        == hashlib.sha256(TARGET_VALUE_SCHEMA_PATH.read_bytes()).hexdigest()
    )
    for metric in benchmark.definition.metrics:
        parameters = {parameter.name: parameter.value for parameter in metric.parameters}
        assert metric.metric_version == "1.1.0-definition"
        assert parameters["scoring_denominator_scope"] == (
            "declared_ordered_2000_feature_panel_only"
        )
        assert parameters["scoring_transform_sha256"] == scoring_sha256
        assert parameters["zero_panel_total_policy"] == (
            "error_fail_evaluation_no_exclusion_or_imputation"
        )


def test_manifest_declares_well_population_semantics_and_h5ad_only_execution(
    manifest: DatasetManifest,
    benchmark: BenchmarkArtifact,
) -> None:
    design = manifest.experimental_design
    assert design.default_split_unit is ExperimentalUnitLevel.PLATE
    assert design.biological_replicate_unit is ExperimentalUnitLevel.WELL
    assert design.randomization_unit is ExperimentalUnitLevel.WELL
    assert design.sampling.subject_unit is ExperimentalUnitLevel.WELL
    assert design.sampling.subject_identity is not None
    assert design.sampling.subject_identity.kind is (
        UnitIdentityExpressionKind.COMPOSITE_SOURCE_FIELDS
    )
    assert design.sampling.subject_identity.source_fields == ("plate", "well")

    contrast = design.randomized_endpoint_contrast
    assert contrast is not None
    predicates = contrast.matched_control.predicates
    assert tuple(predicate.source_field for predicate in predicates) == (
        "dose_value",
        "perturbation",
    )
    assert predicates[0].value_type is ControlPredicateValueType.NUMBER
    assert predicates[0].equals == 0.0
    assert predicates[1].value_type is ControlPredicateValueType.STRING
    assert predicates[1].equals == "control"
    assert contrast.matched_control.stratum_identity.source_fields == ("plate",)

    assert len(benchmark.definition.metrics) == 10
    assert all(
        metric.evaluation_unit.level is ExperimentalUnitLevel.WELL
        for metric in benchmark.definition.metrics
    )
    assert all(
        metric.evaluation_partition_ids == ("p4-untouched-test",)
        for metric in benchmark.definition.metrics
    )
    assert all(
        binding.physical_dataset_binding_id == "sciplex3-k562-24h-physical-source"
        for binding in benchmark.definition.evidence_bindings
    )
    for binding in benchmark.definition.evidence_bindings:
        resolution = manifest.resolve_assessment(
            binding.assessment_reference,
            use_case=DataUseCase.BENCHMARK_EVALUATION,
        )
        assert resolution.data_source_ids == (SOURCE_ID,)
        assert resolution.effective_permission is not None
        assert resolution.effective_permission.status is PermissionStatus.PERMITTED
        assert resolution.use_allowed_without_additional_review


def test_physical_split_and_authoritative_cases_are_exact(
    benchmark: BenchmarkArtifact,
) -> None:
    split = benchmark.definition.split_plan
    case_set = benchmark.definition.evaluation_case_set
    assert split is not None
    assert case_set is not None
    assert len(split.universes) == 1
    assert len(split.protected_group_closures) == 1
    universe = split.universes[0]
    assert universe.record_ids.id_count == 173_652
    assert universe.assignment_unit_ids.id_count == 16

    expected = {
        BenchmarkPartitionRole.TRAIN: (94_785, 768, 8),
        BenchmarkPartitionRole.CALIBRATION: (18_001, 192, 2),
        BenchmarkPartitionRole.MODEL_SELECTION_VALIDATION: (20_481, 192, 2),
        BenchmarkPartitionRole.UNTOUCHED_TEST: (40_385, 384, 4),
    }
    for partition in split.partitions:
        membership = partition.materialized_membership
        assert membership is not None
        expected_records, expected_wells, expected_plates = expected[partition.role]
        assert membership.record_ids.id_count == expected_records
        assert membership.assignment_unit_ids.id_count == expected_plates
        well_membership = next(
            item
            for item in membership.protected_group_memberships
            if item.unit.level is ExperimentalUnitLevel.WELL
        )
        assert well_membership.membership.id_count == expected_wells
        _assert_local_artifact_bytes(membership.record_ids.ids_artifact)
        _assert_local_artifact_bytes(membership.assignment_unit_ids.ids_artifact)
        _assert_local_artifact_bytes(well_membership.membership.ids_artifact)
        _assert_local_artifact_bytes(membership.descendant_closure_artifact)

    assert case_set.case_count == 1_536
    assert case_set.no_action_control_case_count == 32
    assert len(case_set.intervention_case_counts) == 752
    assert {item.case_count for item in case_set.intervention_case_counts} == {2}
    role_counts = Counter(case.role for case in case_set.cases)
    assert role_counts == {
        EvaluationCaseRole.TREATED: 1_504,
        EvaluationCaseRole.MATCHED_NO_ACTION_CONTROL: 32,
    }
    cases_by_unit = {case.evaluation_unit_id: case for case in case_set.cases}
    for context in case_set.contexts:
        context_payload = json.loads(_assert_local_artifact_bytes(context.context_artifact))
        assert "vehicle background" in context_payload["vehicle_background"].casefold()
        assert "immediately before exposure at t=0" in context_payload["population_boundary"]
    assert any(
        "zero modeled active compound" in note.casefold() for note in benchmark.definition.notes
    )
    for case in case_set.cases:
        plate = json.loads(case.evaluation_unit_id)[0]
        assert "t0-assigned-well-population" in case.prediction_subject_id
        if case.role is EvaluationCaseRole.MATCHED_NO_ACTION_CONTROL:
            assert not case.intervention_spec_ids
            assert not case.matched_control_evaluation_unit_ids
            continue
        assert len(case.intervention_spec_ids) == 1
        assert len(case.matched_control_evaluation_unit_ids) == 2
        for control_id in case.matched_control_evaluation_unit_ids:
            assert json.loads(control_id)[0] == plate
            control = cases_by_unit[control_id]
            assert control.role is EvaluationCaseRole.MATCHED_NO_ACTION_CONTROL
            assert control.partition_id == case.partition_id
            assert control.context_id == case.context_id


def test_frozen_component_status_and_typed_baseline_availability(
    benchmark: BenchmarkArtifact,
) -> None:
    definition = benchmark.definition
    assert definition.design_status is BenchmarkLifecycle.FROZEN
    assert definition.intent is BenchmarkIntent.COMPONENT_BENCHMARK
    assert definition.scope.reference_estimand_causal_status is (
        CausalStatus.IDENTIFIED_POPULATION_EFFECT
    )
    assert definition.scope.forecast_causal_status is CausalStatus.PREDICTIVE_ASSOCIATION
    assert benchmark.admission.status is BenchmarkAdmissionStatus.COMPONENT_BENCHMARK
    assert not benchmark.admission.metric_results

    runs = {run.baseline.baseline_id: run for run in benchmark.admission.baseline_runs}
    assert runs["persistence"].status is BaselineRunStatus.NOT_APPLICABLE
    assert runs["temporal-state-space"].status is BaselineRunStatus.NOT_APPLICABLE
    applicable = {
        baseline.baseline_id
        for baseline in definition.baselines
        if baseline.applicability.applies_to(definition.query.state_query)
    }
    assert applicable == {
        "exact-condition-negative-binomial",
        "exact-condition-rep1-empirical-resampling",
        "hierarchical-well-negative-binomial",
        "low-rank-compound-dose-response",
        "matched-vehicle-resampling",
        "nearest-supported-dose",
    }
    assert all(runs[baseline_id].status is BaselineRunStatus.NOT_RUN for baseline_id in applicable)
    assert all(
        isinstance(metric.implementation_binding, SpecificationOnlyImplementationBinding)
        and isinstance(metric.uncertainty.method, SpecificationOnlyImplementationBinding)
        for metric in definition.metrics
    )
    assert all(
        isinstance(baseline.implementation_binding, SpecificationOnlyImplementationBinding)
        for baseline in definition.baselines
    )
    assert any(
        check.status is AuditCheckStatus.NOT_ASSESSED
        for check in benchmark.leakage_audit.checks  # type: ignore[union-attr]
    )


def test_acceptance_policy_uses_paired_baseline_effects_and_absolute_coverage_gates(
    benchmark: BenchmarkArtifact,
) -> None:
    definition = benchmark.definition
    policy = definition.acceptance_policy
    assert policy is not None
    assert len(definition.acceptance_rules) == 12
    groups = {group.group_id: group for group in policy.groups}
    assert groups["all-primary-gates"].operator is AcceptanceGroupOperator.ALL
    assert groups["crps-mandatory-all"].operator is AcceptanceGroupOperator.ALL
    assert groups["joint-effect-noninferior-all"].operator is AcceptanceGroupOperator.ALL
    assert groups["joint-effect-superior-any"].operator is AcceptanceGroupOperator.ANY
    assert groups["coverage-error-all"].operator is AcceptanceGroupOperator.ALL

    crps_rules = tuple(
        rule for rule in definition.acceptance_rules if rule.rule_id.startswith("crps-superior--")
    )
    assert len(crps_rules) == 5
    assert all(
        isinstance(rule.baseline_comparator, ExactBaselineComparator)
        and rule.baseline_margin == 0.0
        and rule.baseline_margin_mode is BaselineMarginMode.ABSOLUTE_DIFFERENCE
        and rule.baseline_requirement is BaselineRequirement.SUPERIOR
        and rule.estimate is ThresholdEstimate.UPPER_CONFIDENCE_BOUND
        for rule in crps_rules
    )

    noninferiority_rules = tuple(
        rule
        for rule in definition.acceptance_rules
        if rule.rule_id.startswith("best-baseline-noninferior--")
    )
    superiority_rules = tuple(
        rule
        for rule in definition.acceptance_rules
        if rule.rule_id.startswith("best-baseline-superior--")
    )
    assert len(noninferiority_rules) == len(superiority_rules) == 2
    assert all(
        isinstance(rule.baseline_comparator, BestApplicableBaselineComparator)
        and rule.baseline_margin == 0.02
        and rule.baseline_margin_mode is BaselineMarginMode.RELATIVE_FRACTION
        and rule.baseline_requirement is BaselineRequirement.NONINFERIOR
        for rule in noninferiority_rules
    )
    assert all(
        isinstance(rule.baseline_comparator, BestApplicableBaselineComparator)
        and rule.baseline_margin == 0.0
        and rule.baseline_margin_mode is BaselineMarginMode.RELATIVE_FRACTION
        and rule.baseline_requirement is BaselineRequirement.SUPERIOR
        for rule in superiority_rules
    )

    coverage_rules = tuple(
        rule
        for rule in definition.acceptance_rules
        if rule.rule_id.startswith("absolute-coverage-error-p")
    )
    assert len(coverage_rules) == 3
    assert all(
        rule.absolute_threshold == 0.03
        and rule.baseline_comparator is None
        and rule.estimate is ThresholdEstimate.UPPER_CONFIDENCE_BOUND
        for rule in coverage_rules
    )
    assert benchmark.admission.paired_baseline_comparisons == ()


def test_adversarial_cell_unit_split_hash_and_cross_plate_control_fail_closed(
    benchmark: BenchmarkArtifact,
) -> None:
    payload = benchmark.model_dump(mode="json")
    payload["definition"]["metrics"][0]["evaluation_unit"] = payload["definition"]["split_plan"][
        "protected_group_closures"
    ][0]["record_unit"]
    with pytest.raises(ValidationError):
        BenchmarkArtifact.model_validate(payload)

    payload = benchmark.model_dump(mode="json")
    membership = payload["definition"]["evaluation_case_set"]["partition_memberships"][0]
    membership["evaluation_unit_ids"]["ids_artifact"]["sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        BenchmarkArtifact.model_validate(payload)

    payload = benchmark.model_dump(mode="json")
    cases = payload["definition"]["evaluation_case_set"]["cases"]
    treated = next(case for case in cases if case["role"] == "treated")
    foreign_control = next(
        case
        for case in cases
        if case["role"] == "matched_no_action_control"
        and case["partition_id"] == treated["partition_id"]
        and case["context_id"] != treated["context_id"]
    )
    treated["matched_control_evaluation_unit_ids"] = sorted(
        [foreign_control["evaluation_unit_id"], treated["matched_control_evaluation_unit_ids"][0]]
    )
    case_bytes = canonical_json_bytes(cases)
    case_artifact = payload["definition"]["evaluation_case_set"]["case_artifact"]
    case_artifact["sha256"] = hashlib.sha256(case_bytes).hexdigest()
    case_artifact["byte_count"] = len(case_bytes)
    with pytest.raises(ValidationError):
        BenchmarkArtifact.model_validate(payload)


def test_component_cannot_be_relabeled_admitted(benchmark: BenchmarkArtifact) -> None:
    payload = benchmark.model_dump(mode="json")
    payload["admission"]["status"] = BenchmarkAdmissionStatus.ADMITTED.value
    with pytest.raises(ValidationError):
        BenchmarkArtifact.model_validate(payload)
