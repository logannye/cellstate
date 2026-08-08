"""Opt-in reference implementations for contract and integration testing."""

from .linear_gaussian import (
    LinearGaussianConfig,
    LinearGaussianPlanner,
    LinearGaussianReference,
    LinearObservationConfig,
    minimal_reference_config,
    sample_posterior,
)

__all__ = [
    "LinearGaussianConfig",
    "LinearGaussianPlanner",
    "LinearGaussianReference",
    "LinearObservationConfig",
    "minimal_reference_config",
    "sample_posterior",
]
