"""Typed failures at capability and contract boundaries."""


class CellStateError(Exception):
    """Base package exception."""


class CapabilityError(CellStateError):
    """The selected backend cannot honestly support part of the request."""


class UnsupportedModalityError(CapabilityError):
    """No observation model exists for an observed modality."""


class UnsupportedInterventionError(CapabilityError):
    """A transition backend cannot represent a recorded intervention."""


class ContractViolationError(CellStateError):
    """A backend returned a result inconsistent with the request."""


class PosteriorCompatibilityError(CellStateError):
    """A serialized posterior cannot be consumed by the selected backend."""
