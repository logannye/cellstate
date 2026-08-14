"""Scientific evaluation primitives."""

from .bootstrap import (
    BootstrapInterval,
    multiway_clustered_bootstrap,
    small_cluster_scale,
    weighted_mean,
)
from .calibration import empirical_interval_coverage, evaluate_marginal_calibration
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
from .query_sufficiency import (
    QuerySufficiencyEvaluator,
    admissible_pre_cutoff_observations,
    evaluate_cohort_sufficiency,
    history_presence_for_cohort,
    request_carries_pre_cutoff_evidence,
)
from .sufficiency import (
    PredictorCapacity,
    evaluate_history_information_gain,
    evaluate_predictive_sufficiency,
    fit_paired_ridge_losses,
    inapplicable_sufficiency_report,
)

__all__ = [
    "METRIC_IMPLEMENTATIONS",
    "BootstrapInterval",
    "MetricImplementation",
    "PredictorCapacity",
    "QuerySufficiencyEvaluator",
    "admissible_pre_cutoff_observations",
    "central_interval",
    "differential_expression_weighted_rmse",
    "effect_rank_agreement",
    "empirical_interval_coverage",
    "energy_score",
    "equal_group_mean",
    "evaluate_cohort_sufficiency",
    "evaluate_history_information_gain",
    "evaluate_marginal_calibration",
    "evaluate_predictive_sufficiency",
    "fit_paired_ridge_losses",
    "history_presence_for_cohort",
    "inapplicable_sufficiency_report",
    "marginal_coverage_error",
    "marginal_interval_coverage",
    "marginal_interval_width",
    "multiway_clustered_bootstrap",
    "profile_rmse",
    "request_carries_pre_cutoff_evidence",
    "sample_crps",
    "small_cluster_scale",
    "weighted_mean",
]
