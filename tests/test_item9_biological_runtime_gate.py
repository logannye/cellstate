"""The experimental bundle contract cannot be bypassed with a hand-built descriptor."""

from __future__ import annotations

import pytest

from cellstate import estimate_cell_state
from cellstate.domain.request import EstimateCellStateRequest
from cellstate.errors import CapabilityError
from cellstate.ports import EstimatorDescriptor, ModelArtifactKind
from cellstate.reference import LinearGaussianReference


class _BiologicalDescriptorWrapper:
    """Delegate implementation with a valid-looking but unauthorised biological identity."""

    def __init__(self, delegate: LinearGaussianReference) -> None:
        self._delegate = delegate

    @property
    def descriptor(self) -> EstimatorDescriptor:
        reference = self._delegate.descriptor
        return EstimatorDescriptor(
            model_id="unregistered-biological-model",
            model_version="1.0.0",
            model_fingerprint=reference.model_fingerprint,
            posterior_schema_id=reference.posterior_schema_id,
            description="Valid-looking descriptor without an executable admission receipt.",
            artifact_kind=ModelArtifactKind.BIOLOGICAL_MODEL,
            support_envelope_id="unresolved-envelope",
            support_envelope_fingerprint="1" * 64,
            training_support_id="unresolved-training",
            training_support_fingerprint="2" * 64,
            validation_evidence_ids=("unresolved-validation",),
            validation_evidence_fingerprints={"unresolved-validation": "3" * 64},
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


def test_biological_descriptor_cannot_bypass_unimplemented_admission_registry(
    model: LinearGaussianReference,
    estimate_request: EstimateCellStateRequest,
) -> None:
    wrapped = _BiologicalDescriptorWrapper(model)
    with pytest.raises(CapabilityError, match="admission registry"):
        estimate_cell_state(estimate_request, estimator=wrapped)  # type: ignore[arg-type]
