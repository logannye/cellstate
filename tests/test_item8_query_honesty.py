from __future__ import annotations

import pytest
from conftest import (
    assay_spec_factory,
    intervention_factory,
    intervention_spec_factory,
    query_factory,
)
from pydantic import ValidationError

from cellstate.domain.common import OntologyTerm, canonical_fingerprint
from cellstate.domain.events import (
    CollectionEffect,
    InterventionEvent,
    InterventionSchedule,
    ObservationCollection,
    ReversibilityStatus,
    ScheduleKind,
)
from cellstate.domain.query import (
    AssayPurpose,
    AssaySpec,
    FutureAssayObservationEndpoint,
    InterventionSpec,
    OutputSpec,
    QueryConstraints,
    StateQuery,
    VersionedReference,
)
from cellstate.reference import LinearGaussianReference, minimal_reference_config


def _protocol(*, fingerprint: str = "6" * 64) -> VersionedReference:
    return VersionedReference(
        reference_id="fixed-endpoint-protocol",
        version="test-v1",
        fingerprint=fingerprint,
    )


def _target_assay(**updates: object) -> AssaySpec:
    payload: dict[str, object] = {
        "assay_id": "fixed-endpoint",
        "modality": OntologyTerm(label="terminal RNA endpoint"),
        "protocol_reference": _protocol(),
        "collection": ObservationCollection(effect=CollectionEffect.TERMINAL_DESTRUCTIVE),
        "purposes": (AssayPurpose.TARGET_ENDPOINT,),
    }
    payload.update(updates)
    return AssaySpec.model_validate(payload)


def test_reversibility_is_typed_and_washout_does_not_invent_it() -> None:
    event = intervention_factory(
        reversibility_status=ReversibilityStatus.UNKNOWN,
    ).model_copy(
        update={
            "schedule": InterventionSchedule(
                kind=ScheduleKind.SINGLE,
                administration_count=1,
                washout_seconds=300,
            )
        }
    )
    action_domain = intervention_spec_factory(
        allowed_reversibility_statuses=(ReversibilityStatus.UNKNOWN,),
    )

    assert action_domain.contains(event)
    assert event.reversibility_status is ReversibilityStatus.UNKNOWN
    assert event.schedule.washout_seconds == 300

    payload = event.model_dump(mode="python")
    payload["reversibility_status"] = True
    with pytest.raises(ValidationError, match="reversibility_status"):
        InterventionEvent.model_validate(payload)

    payload = event.model_dump(mode="python")
    payload.pop("reversibility_status")
    payload["reversible"] = True
    with pytest.raises(ValidationError):
        InterventionEvent.model_validate(payload)

    spec_payload = action_domain.model_dump(mode="python")
    spec_payload["allowed_reversibility_statuses"] = (True,)
    with pytest.raises(ValidationError, match="allowed_reversibility_statuses"):
        InterventionSpec.model_validate(spec_payload)


def test_set_like_query_members_are_canonical_before_fingerprinting() -> None:
    statuses = (
        ReversibilityStatus.UNKNOWN,
        ReversibilityStatus.REVERSIBLE,
        ReversibilityStatus.IRREVERSIBLE,
    )
    first_spec = intervention_spec_factory(allowed_reversibility_statuses=statuses)
    second_spec = intervention_spec_factory(
        allowed_reversibility_statuses=tuple(reversed(statuses))
    )
    assert first_spec.allowed_reversibility_statuses == tuple(ReversibilityStatus)
    assert first_spec == second_spec
    assert canonical_fingerprint(first_spec) == canonical_fingerprint(second_spec)

    assay = assay_spec_factory()
    assay_payload = assay.model_dump(mode="python")
    assay_payload["purposes"] = (
        AssayPurpose.MEASUREMENT_SELECTION,
        AssayPurpose.TARGET_ENDPOINT,
    )
    first_assay = AssaySpec.model_validate(assay_payload)
    assay_payload["purposes"] = tuple(reversed(assay_payload["purposes"]))
    second_assay = AssaySpec.model_validate(assay_payload)
    assert first_assay.purposes == tuple(AssayPurpose)
    assert canonical_fingerprint(first_assay) == canonical_fingerprint(second_assay)


def test_output_value_schema_is_required_separate_and_fingerprinted() -> None:
    base = query_factory()
    output = base.target_outputs[0]
    payload = output.model_dump(mode="python")
    payload.pop("value_schema_reference")
    with pytest.raises(ValidationError, match="value_schema_reference"):
        OutputSpec.model_validate(payload)

    assert hasattr(output.endpoint, "model_reference")
    payload = output.model_dump(mode="python")
    payload["value_schema_reference"] = output.endpoint.model_reference
    with pytest.raises(ValidationError, match="must be distinct"):
        OutputSpec.model_validate(payload)

    changed_output = output.model_copy(
        update={
            "value_schema_reference": output.value_schema_reference.model_copy(
                update={"fingerprint": "0" * 64}
            )
        }
    )
    changed_query = StateQuery.model_validate(
        {**base.model_dump(mode="python"), "target_outputs": (changed_output,)}
    )
    assert changed_output != output
    assert changed_query.fingerprint != base.fingerprint

    compiler = LinearGaussianReference(minimal_reference_config())
    assert compiler.compile(changed_query).query_fingerprint == changed_query.fingerprint
    assert canonical_fingerprint(compiler.compile(changed_query)) != canonical_fingerprint(
        compiler.compile(base)
    )


@pytest.mark.parametrize(
    "economic_sentinel",
    (
        {"cost": 0},
        {"cost_units": "notional"},
        {"turnaround_seconds": 0},
        {"cost": 0, "cost_units": "notional", "turnaround_seconds": 0},
    ),
)
def test_target_only_assays_forbid_economic_sentinels(
    economic_sentinel: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="target-only assays must omit"):
        _target_assay(**economic_sentinel)


def test_measurement_selection_assays_require_complete_economics() -> None:
    target = _target_assay()
    payload = target.model_dump(mode="python")
    payload["purposes"] = (AssayPurpose.MEASUREMENT_SELECTION,)
    with pytest.raises(ValidationError, match="require explicit cost"):
        AssaySpec.model_validate(payload)

    selection = assay_spec_factory()
    for missing_field in ("cost", "cost_units", "turnaround_seconds"):
        incomplete = selection.model_dump(mode="python")
        incomplete[missing_field] = None
        with pytest.raises(ValidationError, match="require explicit cost"):
            AssaySpec.model_validate(incomplete)


def test_future_target_binding_requires_target_endpoint_purpose() -> None:
    base = query_factory()
    output = base.target_outputs[0].model_copy(
        update={
            "endpoint": FutureAssayObservationEndpoint(
                assay_id="fixed-endpoint",
                protocol_reference=_protocol(),
            )
        }
    )
    target_assay = _target_assay()
    no_measurement_budget = base.constraints.model_copy(
        update={
            "maximum_total_assay_cost": None,
            "assay_cost_units": None,
            "maximum_assay_delay_seconds": None,
        }
    )
    query = StateQuery.model_validate(
        {
            **base.model_dump(mode="python"),
            "target_outputs": (output,),
            "available_assays": (target_assay,),
            "constraints": no_measurement_budget,
        }
    )
    assert query.available_assays[0].cost is None
    assert query.constraints.maximum_total_assay_cost is None

    measurement_only = assay_spec_factory(assay_id="fixed-endpoint")
    measurement_only = measurement_only.model_copy(
        update={
            "modality": target_assay.modality,
            "protocol_reference": target_assay.protocol_reference,
        }
    )
    with pytest.raises(ValidationError, match="target-endpoint purpose"):
        StateQuery.model_validate(
            {
                **base.model_dump(mode="python"),
                "target_outputs": (output,),
                "available_assays": (measurement_only,),
            }
        )


def test_query_assay_budget_exists_exactly_for_measurement_selection() -> None:
    base = query_factory()
    with pytest.raises(ValidationError, match="without measurement-selection assays"):
        StateQuery.model_validate({**base.model_dump(mode="python"), "available_assays": ()})

    no_budget = base.constraints.model_copy(
        update={
            "maximum_total_assay_cost": None,
            "assay_cost_units": None,
            "maximum_assay_delay_seconds": None,
        }
    )
    with pytest.raises(ValidationError, match="require explicit assay budget"):
        StateQuery.model_validate({**base.model_dump(mode="python"), "constraints": no_budget})

    constraints_payload = no_budget.model_dump(mode="python")
    constraints_payload["maximum_total_assay_cost"] = 0
    with pytest.raises(ValidationError, match="declared together or all omitted"):
        QueryConstraints.model_validate(constraints_payload)
