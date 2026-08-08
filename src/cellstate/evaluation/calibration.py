"""Small backend-independent uncertainty calibration metrics."""

from __future__ import annotations

from collections.abc import Sequence


def empirical_interval_coverage(
    outcomes: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> float:
    if not outcomes:
        raise ValueError("at least one outcome is required")
    if len(outcomes) != len(lower) or len(outcomes) != len(upper):
        raise ValueError("outcomes and interval bounds must have the same length")
    if any(low > high for low, high in zip(lower, upper, strict=True)):
        raise ValueError("lower interval bounds cannot exceed upper bounds")
    covered = sum(
        low <= value <= high for value, low, high in zip(outcomes, lower, upper, strict=True)
    )
    return covered / len(outcomes)
