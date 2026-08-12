"""Scientific evaluation primitives."""

from .bootstrap import (
    BootstrapInterval,
    multiway_clustered_bootstrap,
    small_cluster_scale,
    weighted_mean,
)
from .calibration import empirical_interval_coverage
from .metrics import (
    METRIC_IMPLEMENTATIONS,
    MetricImplementation,
    central_interval,
    differential_expression_weighted_rmse,
    effect_rank_agreement,
    energy_score,
    equal_group_mean,
    marginal_coverage_error,
    marginal_interval_coverage,
    marginal_interval_width,
    profile_rmse,
    sample_crps,
)
from .sufficiency import evaluate_history_information_gain

__all__ = [
    "METRIC_IMPLEMENTATIONS",
    "BootstrapInterval",
    "MetricImplementation",
    "central_interval",
    "differential_expression_weighted_rmse",
    "effect_rank_agreement",
    "empirical_interval_coverage",
    "energy_score",
    "equal_group_mean",
    "evaluate_history_information_gain",
    "marginal_coverage_error",
    "marginal_interval_coverage",
    "marginal_interval_width",
    "multiway_clustered_bootstrap",
    "profile_rmse",
    "sample_crps",
    "small_cluster_scale",
    "weighted_mean",
]
