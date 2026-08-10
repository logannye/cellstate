from __future__ import annotations

import pytest
from conftest import (
    intervention_factory,
    intervention_spec_factory,
    query_factory,
)
from pydantic import ValidationError
from test_v2_measurement_contracts import (
    _assay_reference,
    _evidence_traces,
    _recommendation,
    _request,
    _supported_evaluation,
)

from cellstate.domain.common import CausalStatus, CriterionOutcome, OntologyTerm, SupportStatus
from cellstate.domain.events import CollectionEffect, ObservationCollection, ReversibilityStatus
from cellstate.domain.measurements import (
    AssayEvaluation,
    MeasurementDecisionRequest,
    MeasurementDecisionStatus,
    MeasurementEvidenceCriterion,
    MeasurementEvidenceTrace,
    MeasurementRecommendation,
)
from cellstate.domain.query import (
    AssayPurpose,
    AssaySpec,
    FutureAssayObservationEndpoint,
    OutputSpec,
    StateQuery,
    VersionedReference,
    VersionedTransformEndpoint,
)


def _reference(reference_id: str, fingerprint: str) -> VersionedReference:
    return VersionedReference(
        reference_id=reference_id,
        version="test-v1",
        fingerprint=fingerprint,
    )


def test_reversibility_status_is_an_exact_action_domain_boundary() -> None:
    """Washout metadata must not make UNKNOWN reversibility interchangeable with reversible."""

    specification = intervention_spec_factory(
        allowed_reversibility_statuses=(ReversibilityStatus.UNKNOWN,),
    )
    unknown = intervention_factory(reversibility_status=ReversibilityStatus.UNKNOWN)
    reversible = intervention_factory(reversibility_status=ReversibilityStatus.REVERSIBLE)
    query = StateQuery.model_validate(
        {
            **query_factory().model_dump(mode="python"),
            "intervention_space": (specification,),
        }
    )

    assert specification.contains(unknown)
    assert not specification.contains(reversible)
    assert query.contains_intervention(unknown)
    assert not query.contains_intervention(reversible)
    assert query.realization_evidence_gaps(reversible, {}) == (
        "intervention_outside_declared_action_domain",
    )


@pytest.mark.parametrize("endpoint_kind", ("future_assay", "versioned_transform"))
def test_value_schema_cannot_alias_any_observation_or_transform_contract(
    endpoint_kind: str,
) -> None:
    output = query_factory().target_outputs[0]
    if endpoint_kind == "future_assay":
        endpoint_reference = _reference("future-endpoint-protocol", "1" * 64)
        endpoint = FutureAssayObservationEndpoint(
            assay_id="future-endpoint",
            protocol_reference=endpoint_reference,
        )
    else:
        endpoint_reference = _reference("derived-output-transform", "2" * 64)
        endpoint = VersionedTransformEndpoint(
            source_term=OntologyTerm(label="raw terminal readout"),
            transformation_reference=endpoint_reference,
        )

    payload = output.model_dump(mode="python")
    payload.update(
        {
            "endpoint": endpoint,
            "value_schema_reference": endpoint_reference,
        }
    )
    with pytest.raises(ValidationError, match="value-schema reference must be distinct"):
        OutputSpec.model_validate(payload)


@pytest.mark.parametrize(
    ("assay_update", "message"),
    (
        ({"cost_units": "USD"}, "cost units must match"),
        ({"cost": 101.0}, "exceeds the query cost"),
        ({"turnaround_seconds": 100_001.0}, "exceeds the query delay"),
        (
            {
                "collection": ObservationCollection(
                    effect=CollectionEffect.PARTIALLY_DESTRUCTIVE_POPULATION_SAMPLING,
                    sampling_fraction=0.1,
                )
            },
            "partial population assays require an aggregate query subject",
        ),
    ),
)
def test_measurement_selection_assays_must_fit_the_exact_query_budget_and_subject(
    assay_update: dict[str, object],
    message: str,
) -> None:
    base = query_factory()
    assay_payload = base.available_assays[0].model_dump(mode="python")
    assay_payload.update(assay_update)
    assay = AssaySpec.model_validate(assay_payload)

    with pytest.raises(ValidationError, match=message):
        StateQuery.model_validate(
            {
                **base.model_dump(mode="python"),
                "available_assays": (assay,),
            }
        )


def test_target_endpoint_assays_bind_protocol_and_cannot_be_declared_unused() -> None:
    base = query_factory()
    selection_assay = base.available_assays[0]
    combined_assay = AssaySpec.model_validate(
        {
            **selection_assay.model_dump(mode="python"),
            "purposes": (
                AssayPurpose.TARGET_ENDPOINT,
                AssayPurpose.MEASUREMENT_SELECTION,
            ),
        }
    )

    with pytest.raises(ValidationError, match="must be referenced by a query target"):
        StateQuery.model_validate(
            {
                **base.model_dump(mode="python"),
                "available_assays": (combined_assay,),
            }
        )

    output = base.target_outputs[0].model_copy(
        update={
            "endpoint": FutureAssayObservationEndpoint(
                assay_id=combined_assay.assay_id,
                protocol_reference=_reference("wrong-protocol", "3" * 64),
            )
        }
    )
    with pytest.raises(ValidationError, match="protocol does not match"):
        StateQuery.model_validate(
            {
                **base.model_dump(mode="python"),
                "target_outputs": (output,),
                "available_assays": (combined_assay,),
            }
        )


def test_measurement_request_rejects_ambiguous_or_unpriced_decision_sets() -> None:
    request = _request()

    payload = request.model_dump(mode="python")
    payload["candidates"] = (payload["candidates"][0], payload["candidates"][0])
    with pytest.raises(ValidationError, match="candidate scenario IDs must be unique"):
        MeasurementDecisionRequest.model_validate(payload)

    payload = request.model_dump(mode="python")
    payload["candidates"][1]["horizon_name"] = "undeclared-horizon"
    with pytest.raises(ValidationError, match="must use the objective horizon"):
        MeasurementDecisionRequest.model_validate(payload)

    payload = request.model_dump(mode="python")
    payload["candidates"][0]["subject"]["subject_id"] = "different-cell"
    with pytest.raises(ValidationError, match="must use one typed subject"):
        MeasurementDecisionRequest.model_validate(payload)

    payload = request.model_dump(mode="python")
    payload["destructiveness_penalties"][
        CollectionEffect.VIABILITY_PRESERVING_WITH_KNOWN_EFFECT
    ] = -0.1
    with pytest.raises(ValidationError, match="finite and nonnegative"):
        MeasurementDecisionRequest.model_validate(payload)

    payload = request.model_dump(mode="python")
    payload["destructiveness_penalties"][CollectionEffect.NONDESTRUCTIVE] = 0.1
    with pytest.raises(ValidationError, match="nondestructive collection must have zero"):
        MeasurementDecisionRequest.model_validate(payload)


def test_measurement_evidence_traces_are_content_addressed_and_nonvacuous() -> None:
    trace = MeasurementEvidenceTrace(
        criterion=MeasurementEvidenceCriterion.ASSAY_OUTCOME_MODEL,
        outcome=CriterionOutcome.PASSED,
        scope_fingerprint="3" * 64,
        evidence_ids=("assay-validation",),
        evidence_fingerprints={"assay-validation": "a" * 64},
    )

    payload = trace.model_dump(mode="python")
    payload["evidence_ids"] = (" ",)
    payload["evidence_fingerprints"] = {" ": "a" * 64}
    with pytest.raises(ValidationError, match="evidence IDs must be nonblank"):
        MeasurementEvidenceTrace.model_validate(payload)

    payload = trace.model_dump(mode="python")
    payload["evidence_ids"] = ("assay-validation", "assay-validation")
    with pytest.raises(ValidationError, match="evidence IDs must be unique"):
        MeasurementEvidenceTrace.model_validate(payload)

    payload = trace.model_dump(mode="python")
    payload["evidence_fingerprints"] = {"different-artifact": "a" * 64}
    with pytest.raises(ValidationError, match="one fingerprint per artifact"):
        MeasurementEvidenceTrace.model_validate(payload)

    payload = trace.model_dump(mode="python")
    payload["evidence_fingerprints"] = {"assay-validation": "not-a-sha256"}
    with pytest.raises(ValidationError, match="must be SHA-256"):
        MeasurementEvidenceTrace.model_validate(payload)

    payload = trace.model_dump(mode="python")
    payload["reasons"] = (" ",)
    with pytest.raises(ValidationError, match="reasons must be nonblank"):
        MeasurementEvidenceTrace.model_validate(payload)

    with pytest.raises(ValidationError, match="failed measurement criterion requires evidence"):
        MeasurementEvidenceTrace(
            criterion=MeasurementEvidenceCriterion.DECISION_UTILITY,
            outcome=CriterionOutcome.FAILED,
            scope_fingerprint="3" * 64,
        )


def test_numeric_assay_evaluations_fail_closed_without_complete_support() -> None:
    supported = _supported_evaluation()

    payload = supported.model_dump(mode="python")
    payload["expected_information_gain"] = None
    with pytest.raises(ValidationError, match="requires every value and metric"):
        AssayEvaluation.model_validate(payload)

    payload = supported.model_dump(mode="python")
    payload["measurement_model_evidence_ids"] = ()
    payload["measurement_model_evidence_fingerprints"] = {}
    with pytest.raises(ValidationError, match="requires measurement-model validation evidence"):
        AssayEvaluation.model_validate(payload)

    payload = supported.model_dump(mode="python")
    payload["reasons"] = ("unsupported despite numeric values",)
    with pytest.raises(ValidationError, match="cannot contain support blockers"):
        AssayEvaluation.model_validate(payload)

    payload = supported.model_dump(mode="python")
    payload["measurement_model_evidence_ids"] = ("different-model",)
    payload["measurement_model_evidence_fingerprints"] = {"different-model": "b" * 64}
    with pytest.raises(ValidationError, match="must match the assay-outcome evidence trace"):
        AssayEvaluation.model_validate(payload)

    unavailable = AssayEvaluation(
        assay=_assay_reference(),
        status=SupportStatus.UNSUPPORTED,
        collection_effect=CollectionEffect.NONDESTRUCTIVE,
        evidence_traces=_evidence_traces(outcome=CriterionOutcome.UNSUPPORTED),
        reasons=("assay outcome model is outside validated support",),
    )
    assert unavailable.canonical_net_decision_value is None

    payload = unavailable.model_dump(mode="python")
    payload["measurement_model_evidence_ids"] = ("unsupported-model",)
    payload["measurement_model_evidence_fingerprints"] = {"unsupported-model": "c" * 64}
    with pytest.raises(ValidationError, match="cannot claim measurement-model evidence"):
        AssayEvaluation.model_validate(payload)

    payload = unavailable.model_dump(mode="python")
    payload["reasons"] = ()
    with pytest.raises(ValidationError, match="requires explicit reasons"):
        AssayEvaluation.model_validate(payload)


def test_measurement_recommendation_binds_exact_request_and_support_evidence() -> None:
    recommendation = _recommendation()

    payload = recommendation.model_dump(mode="python")
    payload["candidates"] = (payload["candidates"][0], payload["candidates"][0])
    with pytest.raises(ValidationError, match="candidate scenario IDs must be unique"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload["evaluations"][0]["assay"]["fingerprint"] = "9" * 64
    with pytest.raises(ValidationError, match="references must match the requested assays"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload["provenance"]["query_fingerprint"] = "9" * 64
    with pytest.raises(ValidationError, match="provenance/query fingerprints must agree"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload["causal_status"] = CausalStatus.TRANSPORTED_UNDER_ASSUMPTIONS
    with pytest.raises(ValidationError, match="causal status must match"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload["readiness"]["causal"] = CriterionOutcome.FAILED
    payload["readiness"]["valid_for_control"] = False
    with pytest.raises(ValidationError, match="causal support must match readiness"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    trace = payload["evaluations"][0]["evidence_traces"][1]
    trace["evidence_ids"] = ("assay-validation", "unrecorded-evidence")
    trace["evidence_fingerprints"] = {
        "assay-validation": "a" * 64,
        "unrecorded-evidence": "d" * 64,
    }
    with pytest.raises(ValidationError, match="must cite validation artifacts"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    evaluation = payload["evaluations"][0]
    evaluation["measurement_model_evidence_ids"] = ("unrecorded-model",)
    evaluation["measurement_model_evidence_fingerprints"] = {"unrecorded-model": "e" * 64}
    evaluation["evidence_traces"][0]["evidence_ids"] = ("unrecorded-model",)
    evaluation["evidence_traces"][0]["evidence_fingerprints"] = {"unrecorded-model": "e" * 64}
    with pytest.raises(ValidationError, match="must cite validation artifacts"):
        MeasurementRecommendation.model_validate(payload)


def test_supported_measurement_values_cannot_outrun_readiness_or_selection_state() -> None:
    recommendation = _recommendation()

    payload = recommendation.model_dump(mode="python")
    payload["readiness"]["measurement_model"] = CriterionOutcome.UNSUPPORTED
    payload["readiness"]["valid_for_measurement_selection"] = False
    with pytest.raises(ValidationError, match="not scientifically ready"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload["selected_assay_id"] = None
    with pytest.raises(ValidationError, match="requires a selected assay"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload["abstention_reasons"] = ("also abstain",)
    with pytest.raises(ValidationError, match="cannot also report abstention reasons"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload["selected_assay_id"] = "unrequested-assay"
    with pytest.raises(ValidationError, match="selected assay must have a supported"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload.update(
        {
            "status": MeasurementDecisionStatus.ABSTAINED,
            "selected_assay_id": recommendation.selected_assay_id,
            "abstention_reasons": ("attempted abstention",),
        }
    )
    with pytest.raises(ValidationError, match="cannot select an assay"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload.update(
        {
            "status": MeasurementDecisionStatus.ABSTAINED,
            "selected_assay_id": None,
            "abstention_reasons": (),
        }
    )
    with pytest.raises(ValidationError, match="requires explicit reasons"):
        MeasurementRecommendation.model_validate(payload)

    payload = recommendation.model_dump(mode="python")
    payload.update(
        {
            "status": MeasurementDecisionStatus.ABSTAINED,
            "selected_assay_id": None,
            "abstention_reasons": ("value ignored",),
        }
    )
    with pytest.raises(ValidationError, match="above the declared threshold"):
        MeasurementRecommendation.model_validate(payload)
