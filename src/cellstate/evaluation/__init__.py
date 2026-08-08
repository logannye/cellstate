"""Scientific evaluation primitives."""

from .calibration import empirical_interval_coverage
from .sufficiency import evaluate_history_information_gain

__all__ = ["empirical_interval_coverage", "evaluate_history_information_gain"]
