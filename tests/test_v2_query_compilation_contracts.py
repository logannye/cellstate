from __future__ import annotations

import pytest
from conftest import (
    SYNTHETIC_TEST_OPTIONS,
    environment_spec_factory,
    intervention_factory,
    intervention_spec_factory,
    observation_factory,
    query_factory,
    request_factory,
    subject_factory,
)
from pydantic import ValidationError

from cellstate.api import estimate_cell_state
from cellstate.domain.common import OntologyTerm
from cellstate.domain.events import (
    ActualPerturbation,
    AssignmentMechanism,
    MatchedControl,
    PerturbationStatus,
)
from cellstate.domain.query import (
    FutureAssayObservationEndpoint,
    MissingHistoryPolicy,
    PrecisionRequirement,
    PredictionHorizon,
    StateQuery,
    Timescale,
    VersionedReference,
)
from cellstate.domain.scenarios import (
    InterventionObjective,
    ObjectiveDirection,
    ObjectiveTerm,
)
from cellstate.domain.specification import CompiledStateSpecification
from cellstate.reference import (
    LinearGaussianPlanner,
    LinearGaussianReference,
    minimal_reference_config,
)


def _reference() -> VersionedReference:
    return VersionedReference(
        reference_id="query-contract-test-reference",
        version="1.0.0",
        fingerprint="a" * 64,
    )


def test_query_requires_a_positive_finite_usable_temporal_resolution() -> None:
    payload = query_factory().model_dump(mode="python")
    payload.pop("temporal_resolution_seconds")
    with pytest.raises(ValidationError, match="temporal_resolution_seconds"):
        StateQuery.model_validate(payload)

    payload["temporal_resolution_seconds"] = 0
    with pytest.raises(ValidationError, match="greater than 0"):
        StateQuery.model_validate(payload)

    payload["temporal_resolution_seconds"] = 61
    with pytest.raises(ValidationError, match="cannot exceed a prediction horizon"):
        StateQuery.model_validate(payload)


def test_target_horizons_and_future_assay_protocols_are_query_bounded() -> None:
    query = query_factory()
    target = query.target_outputs[0].model_copy(update={"supported_horizon_names": ("undeclared",)})
    with pytest.raises(ValidationError, match="undeclared supported horizons"):
        StateQuery.model_validate({**query.model_dump(mode="python"), "target_outputs": (target,)})

    future_assay_target = query.target_outputs[0].model_copy(
        update={
            "endpoint": FutureAssayObservationEndpoint(
                assay_id="undeclared-assay",
                protocol_reference=_reference(),
            )
        }
    )
    with pytest.raises(ValidationError, match="undeclared future assay"):
        StateQuery.model_validate(
            {**query.model_dump(mode="python"), "target_outputs": (future_assay_target,)}
        )


def test_precision_requirement_must_use_a_target_supported_horizon() -> None:
    query = query_factory()
    intermediate = PredictionHorizon(
        name="intermediate",
        duration_seconds=3_600,
        timescale=Timescale.INTERMEDIATE,
    )
    requirement = PrecisionRequirement(
        target=query.target_outputs[0].term,
        horizon_name="intermediate",
        metric="absolute_error",
        maximum_error=0.1,
        units=query.target_outputs[0].units,
    )
    with pytest.raises(ValidationError, match="horizon unsupported by its target"):
        StateQuery.model_validate(
            {
                **query.model_dump(mode="python"),
                "prediction_horizons": (*query.prediction_horizons, intermediate),
                "precision_requirements": (requirement,),
            }
        )


@pytest.mark.parametrize(
    "mutation",
    ("target", "horizon", "intervention", "environment", "evidence", "constraints", "resolution"),
)
def test_compiled_specification_content_binds_every_query_semantic(mutation: str) -> None:
    query = query_factory()
    model = LinearGaussianReference(minimal_reference_config())
    specification = model.compile(query)
    payload = specification.model_dump(mode="python")

    if mutation == "target":
        payload["target_outputs"][0]["weight"] = 2
    elif mutation == "horizon":
        payload["prediction_horizons"][0]["duration_seconds"] = 59
    elif mutation == "intervention":
        payload["intervention_space"][0]["dose_domain"]["maximum"] = 99
    elif mutation == "environment":
        payload["environment_space"] = (environment_spec_factory(required=False),)
    elif mutation == "evidence":
        payload["evidence_policy"]["minimum_observed_measurements"] = 1
    elif mutation == "constraints":
        payload["constraints"]["allow_transport"] = True
    elif mutation == "resolution":
        payload["temporal_resolution_seconds"] = 2
    else:  # pragma: no cover - parameter table is closed above
        raise AssertionError(mutation)

    with pytest.raises(ValidationError, match="exactly reproduce"):
        CompiledStateSpecification.model_validate(payload)


def test_compiled_factors_cannot_name_undeclared_required_outputs() -> None:
    model = LinearGaussianReference(minimal_reference_config())
    payload = model.compile(query_factory()).model_dump(mode="python")
    payload["active_factors"][0]["required_for_outputs"] = ("undeclared-output",)
    with pytest.raises(ValidationError, match="undeclared target outputs"):
        CompiledStateSpecification.model_validate(payload)


def test_action_membership_enforces_assignment_randomization_and_matched_control() -> None:
    base = intervention_spec_factory()
    strict_spec = type(base).model_validate(
        {
            **base.model_dump(mode="python"),
            "allowed_assignment_mechanisms": (AssignmentMechanism.RANDOMIZED,),
            "randomization_unit_kind": "well",
            "require_randomization_unit": True,
            "require_matched_control": True,
        }
    )
    subject = subject_factory()
    control = MatchedControl(
        subject_id="control-cell",
        assignment_unit_id="control-well",
        condition=OntologyTerm(label="vehicle control"),
        matching_basis="same plate, batch, medium, and collection time",
        contemporaneous=True,
    )
    event = intervention_factory(
        subject=subject,
        assignment_mechanism=AssignmentMechanism.RANDOMIZED,
        assignment_unit_kind="well",
        assignment_unit_id=subject.experimental_unit_id,
        randomization_unit_kind="well",
        randomization_unit_id="randomization-block-1",
        matched_control=control,
    )

    assert strict_spec.contains(event)
    assert not strict_spec.contains(
        event.model_copy(update={"assignment_mechanism": AssignmentMechanism.OBSERVATIONAL})
    )
    assert not strict_spec.contains(event.model_copy(update={"assignment_unit_kind": "cell"}))
    assert not strict_spec.contains(event.model_copy(update={"randomization_unit_kind": "plate"}))
    assert not strict_spec.contains(event.model_copy(update={"matched_control": None}))


def test_randomized_events_require_complete_distinct_unit_links() -> None:
    payload = intervention_factory().model_dump(mode="python")
    payload.update(
        {
            "assignment_mechanism": AssignmentMechanism.RANDOMIZED,
            "randomization_unit_kind": None,
            "randomization_unit_id": None,
        }
    )
    with pytest.raises(ValidationError, match="requires an explicit randomization unit"):
        type(intervention_factory()).model_validate(payload)

    payload["randomization_unit_kind"] = "well"
    with pytest.raises(ValidationError, match="declared together"):
        type(intervention_factory()).model_validate(payload)

    payload["randomization_unit_id"] = "block-1"
    payload["matched_control"] = MatchedControl(
        subject_id=subject_factory().subject_id,
        assignment_unit_id="control-well",
        condition=OntologyTerm(label="vehicle control"),
        matching_basis="same batch",
        contemporaneous=True,
    )
    with pytest.raises(ValidationError, match="distinct biological subject"):
        type(intervention_factory()).model_validate(payload)


def test_realization_evidence_gaps_preserve_uncertain_actions() -> None:
    query = query_factory()
    event = intervention_factory(actual_perturbation=None)
    observation = observation_factory(event_id="target-engagement")
    assert query.realization_evidence_gaps(event, {observation.event_id: observation}) == (
        "realization_not_assessed",
    )

    uncertain = event.model_copy(
        update={
            "actual_perturbation": ActualPerturbation(
                status=PerturbationStatus.UNKNOWN,
                evidence_event_ids=(observation.event_id,),
            )
        }
    )
    assert (
        query.realization_evidence_gaps(
            uncertain,
            {observation.event_id: observation},
        )
        == ()
    )


def test_reference_rejects_unimplemented_environment_missingness_policies() -> None:
    query = query_factory()
    environment = environment_spec_factory(required=True).model_copy(
        update={"missing_history_policy": MissingHistoryPolicy.REPRESENT_AS_UNKNOWN}
    )
    query = StateQuery.model_validate(
        {**query.model_dump(mode="python"), "environment_space": (environment,)}
    )
    request = request_factory(query=query)
    model = LinearGaussianReference(minimal_reference_config())
    specification = model.compile(query)

    report = model.capabilities(request, specification)
    assert not report.supported
    assert any(
        "missing-policy=represent_as_unknown" in item for item in report.unsupported_environments
    )


def test_planner_capabilities_respect_each_targets_supported_horizons() -> None:
    query = query_factory()
    query = StateQuery.model_validate(
        {
            **query.model_dump(mode="python"),
            "prediction_horizons": (
                *query.prediction_horizons,
                PredictionHorizon(
                    name="intermediate",
                    duration_seconds=3_600,
                    timescale=Timescale.INTERMEDIATE,
                ),
            ),
        }
    )
    model = LinearGaussianReference(minimal_reference_config())
    belief = estimate_cell_state(
        request_factory(query=query),
        estimator=model,
        options=SYNTHETIC_TEST_OPTIONS,
    )
    target = query.target_outputs[0].term
    objective = InterventionObjective(
        objective_id="unsupported-target-horizon-pair",
        horizon_name="intermediate",
        terms=(ObjectiveTerm(target=target, direction=ObjectiveDirection.MAXIMIZE),),
    )

    report = LinearGaussianPlanner(model).capabilities(belief, objective, ())
    assert not report.supported
    assert report.unsupported_outputs == (target.key,)
