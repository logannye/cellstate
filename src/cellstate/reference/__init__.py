"""Opt-in reference implementations for contract and integration testing."""

from .linear_gaussian import (
    LinearGaussianConfig,
    LinearGaussianMeasurementPolicy,
    LinearGaussianPlanner,
    LinearGaussianReference,
    LinearObservationConfig,
    minimal_reference_config,
    sample_posterior,
)

__all__ = [
    "LinearGaussianConfig",
    "LinearGaussianMeasurementPolicy",
    "LinearGaussianPlanner",
    "LinearGaussianReference",
    "LinearObservationConfig",
    "minimal_reference_config",
    "sample_posterior",
]
