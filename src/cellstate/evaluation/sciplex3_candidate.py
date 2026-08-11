"""Frozen p1-only sci-Plex3 Gamma--Poisson factor candidate.

This module is deliberately narrower than a cell-state backend.  It fits and samples one
query-bound population assay-response distribution and never constructs a ``CellStateBelief``.
Only p1 replicate-1 rows may enter fitting.  Protected-partition admission and calibration remain
external trusted-workflow responsibilities.

The candidate is a continuous, non-categorical, rank-16 Gamma--Poisson admixture:

``B_k ~ Dirichlet(0.3)``, ``theta_k ~ Gamma(r_theta, r_theta / m[action, plate, k])``,
and ``y_g ~ Poisson(sum_k theta_k B[k, g])``.  The factor shape is fixed at exactly
``r_theta = 0.1``; it is never estimated from p1.

Every loading row sums to one, so the model generates a future panel total.  It never conditions
on an observed target depth.  New plates use one conservative, deterministic unit context rather
than replaying a concentrated observed p1 ``rho`` row.  Observable counts are sampled exactly
conditional on a positive panel through the source-free v5 compound-Poisson construction.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Final, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import svds
from scipy.special import digamma, gammaln

from cellstate.evaluation.sciplex3_baselines import (
    NO_ACTION,
    RNG_ALGORITHM,
    SCIPLEX3_FEATURE_COUNT,
    SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256,
    CompoundDose,
    NoAction,
    P1TrainingData,
    P1WellCounts,
    PredictionTarget,
    TargetCondition,
)
from cellstate.evaluation.sciplex3_candidate_v5 import (
    SCIPLEX3_V5_EQUAL_WELL_SCALE,
    SCIPLEX3_V5_GRADIENT_TOL,
    SCIPLEX3_V5_OBJECTIVE_VERSION,
    SciPlex3V5ActionParameters,
    SciPlex3V5Design,
    SciPlex3V5ObjectiveError,
    fit_fixed_q_action_context_m_step,
    fixed_q_full_elbo_action_context,
)
from cellstate.evaluation.sciplex3_sampling_v5 import (
    SCIPLEX3_V5_MAX_COMPOUND_POISSON_INTENSITY,
    SCIPLEX3_V5_MAX_SAMPLE_COUNT,
    SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG,
    SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256,
    SciPlex3SamplingV5Error,
    V5PositiveConditionedSampler,
    V5SampleRequest,
    V5SamplingEnvelopeCertificate,
    V5SamplingParameters,
    V5SamplingTarget,
    canonical_target_fingerprint,
    freeze_positive_int64_samples,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SCIPLEX3_CANDIDATE_MODEL_ID: Final = (
    "sciplex3-k562-24h-gamma-poisson-fixed-r0p1-neutral-unseen-plate-k16-v5"
)
SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION: Final = "5.0.0"
SCIPLEX3_CANDIDATE_MODEL_SCHEMA: Final = (
    "sciplex3-gamma-poisson-fixed-r0p1-neutral-context-candidate-model-v5"
)
SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION: Final = "5.0.0"
SCIPLEX3_CANDIDATE_SPECIFICATION_SCHEMA: Final = (
    "sciplex3-gamma-poisson-fixed-r0p1-neutral-context-candidate-specification-v5"
)
SCIPLEX3_CANDIDATE_SPECIFICATION_SCHEMA_VERSION: Final = "5.0.0"
SCIPLEX3_CANDIDATE_OUTPUT_MODEL_TOPOLOGY: Final = (
    "sciplex3-gamma-poisson-fixed-r0p1-neutral-context-output-topology-v5"
)
SCIPLEX3_CANDIDATE_FACTOR_COUNT: Final = 16
SCIPLEX3_CANDIDATE_COMPOUND_COUNT: Final = 188
SCIPLEX3_CANDIDATE_PLATE_COUNT: Final = 8
SCIPLEX3_CANDIDATE_ACTION_COUNT: Final = 752
SCIPLEX3_CANDIDATE_DOSES_NM: Final = (10, 100, 1_000, 10_000)
SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT: Final = 94_785
SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT: Final = 768
SCIPLEX3_CANDIDATE_CONTROL_WELL_COUNT: Final = 16
SCIPLEX3_CANDIDATE_ZERO_PANEL_RECORD_COUNT: Final = 7
SCIPLEX3_CANDIDATE_DIRICHLET_CONCENTRATION: Final = 0.3
SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE: Final = 0.1
SCIPLEX3_CANDIDATE_FACTOR_SHAPE_LOWER_GUARD: Final = 0.05
SCIPLEX3_CANDIDATE_DOSE_MAGNITUDE_SD: Final = 2.0
SCIPLEX3_CANDIDATE_DOSE_SECOND_DIFFERENCE_SD: Final = 1.0
SCIPLEX3_CANDIDATE_BATCH_SIZE: Final = 512
SCIPLEX3_CANDIDATE_MIN_OUTER_ITERATIONS: Final = 10
SCIPLEX3_CANDIDATE_MAX_OUTER_ITERATIONS: Final = 50
SCIPLEX3_CANDIDATE_CONVERGENCE_RTOL: Final = 1e-7
SCIPLEX3_CANDIDATE_CONVERGENCE_STREAK: Final = 3
SCIPLEX3_CANDIDATE_ELBO_DECREASE_RTOL: Final = 1e-8
SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL: Final = 1e-8
SCIPLEX3_CANDIDATE_INNER_CONVERGENCE_STREAK: Final = 2
SCIPLEX3_CANDIDATE_MIN_INNER_SWEEPS: Final = 2
SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS: Final = 50
SCIPLEX3_CANDIDATE_MASS_EPS_MULTIPLIER: Final = 64.0
SCIPLEX3_CANDIDATE_SVD_BOUNDARY_TIE_RTOL: Final = 1e-12
SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS: Final = 12
SCIPLEX3_CANDIDATE_IDENTIFIABILITY_THRESHOLD: Final = math.sqrt(np.finfo(np.float64).eps)
SCIPLEX3_CANDIDATE_IDENTIFIABILITY_MARGIN: Final = 1e-12
SCIPLEX3_CANDIDATE_QUANTIZATION_BOUNDARY_EPS_MULTIPLIER: Final = 64.0
SCIPLEX3_CANDIDATE_LAMBDA_INITIAL_MASS: Final = 1_000.0
SCIPLEX3_CANDIDATE_NNDSVD_SCORE_FLOOR_FRACTION: Final = 1e-8
SCIPLEX3_CANDIDATE_TARGET_PARTITIONS: Final = (
    "p2-calibration",
    "p3-model-selection-validation",
    "p4-untouched-test",
)
SCIPLEX3_CANDIDATE_CONTEXT_ID: Final = "sciplex3-k562-24h-population-assay-response-v1"
SCIPLEX3_CANDIDATE_TAU_GRID: Final = tuple(math.exp(index / 20.0) for index in range(-20, 7))
SCIPLEX3_CANDIDATE_V5_NEUTRAL_CONTEXT_ID: Final = "neutral-unit-unseen-plate-context"
SCIPLEX3_CANDIDATE_V5_NO_ACTION_ID: Final = "no-action"
SCIPLEX3_CANDIDATE_REFERENCE_RUNTIME: Final[Mapping[str, object]] = MappingProxyType(
    {
        "blas_name": "scipy-openblas",
        "blas_version": "0.3.31.188.0",
        "byte_order": "little",
        "numpy_version": "2.4.6",
        "platform_machine": "x86_64",
        "python_implementation": "CPython",
        "python_version": "3.11.15",
        "scipy_version": "1.17.1",
        "single_thread": True,
    }
)


class SciPlex3CandidateError(ValueError):
    """Raised when candidate fitting, loading, or sampling violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class CandidateSampleRequest:
    """Seed-bound, outcome-free request whose complete count enters v5 support."""

    target: PredictionTarget
    sample_count: int
    seed: int

    def __post_init__(self) -> None:
        if type(self.target) is not PredictionTarget:
            raise SciPlex3CandidateError(
                "candidate request target must be an exact PredictionTarget"
            )
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise SciPlex3CandidateError("candidate sample_count must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed <= np.iinfo(np.uint64).max:
            raise SciPlex3CandidateError("candidate seed must be an unsigned 64-bit integer")


@dataclass(frozen=True, slots=True, eq=False)
class CandidateRawCountSamples:
    """Immutable candidate samples with exact model/calibration/contract provenance."""

    candidate_id: str
    target: PredictionTarget
    ordered_feature_keys: tuple[str, ...]
    model_artifact_sha256: str
    calibration_state_sha256: str
    sampling_contract_sha256: str
    target_fingerprint: str
    context_id: str
    seed: int
    samples: IntArray
    rng_algorithm: Literal["numpy-pcg64dxsm-v1"] = RNG_ALGORITHM

    def __post_init__(self) -> None:
        if self.candidate_id != SCIPLEX3_CANDIDATE_MODEL_ID:
            raise SciPlex3CandidateError("candidate sample provenance has the wrong model ID")
        if type(self.target) is not PredictionTarget:
            raise SciPlex3CandidateError("candidate sample target has the wrong exact type")
        keys = tuple(self.ordered_feature_keys)
        if len(keys) != SCIPLEX3_FEATURE_COUNT or len(set(keys)) != len(keys):
            raise SciPlex3CandidateError("candidate sample feature panel is invalid")
        if type(self.seed) is not int or not 0 <= self.seed <= np.iinfo(np.uint64).max:
            raise SciPlex3CandidateError("candidate sample seed is invalid")
        if self.rng_algorithm != RNG_ALGORITHM:
            raise SciPlex3CandidateError("candidate sample RNG algorithm drifted")
        for value, name in (
            (self.model_artifact_sha256, "candidate sample model artifact sha256"),
            (self.calibration_state_sha256, "candidate sample calibration state sha256"),
            (self.sampling_contract_sha256, "candidate sample contract sha256"),
            (self.target_fingerprint, "candidate sample target fingerprint"),
        ):
            _strict_sha256(value, name=name)
        if self.sampling_contract_sha256 != SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256:
            raise SciPlex3CandidateError("candidate sample contract provenance drifted")
        if self.target_fingerprint != _v5_target_fingerprint(self.target):
            raise SciPlex3CandidateError("candidate sample target provenance drifted")
        if self.context_id != SCIPLEX3_CANDIDATE_V5_NEUTRAL_CONTEXT_ID:
            raise SciPlex3CandidateError("candidate sample context is not the neutral v5 context")
        try:
            frozen = freeze_positive_int64_samples(self.samples)
        except SciPlex3SamplingV5Error as error:
            raise SciPlex3CandidateError("candidate raw-count samples are invalid") from error
        if frozen.shape[1] != SCIPLEX3_FEATURE_COUNT:
            raise SciPlex3CandidateError("candidate raw-count sample feature panel is invalid")
        object.__setattr__(self, "ordered_feature_keys", keys)
        object.__setattr__(self, "samples", frozen)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SciPlex3CandidateError(
            "candidate payload is not canonical-JSON compatible"
        ) from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _v5_action_id(condition: TargetCondition) -> str:
    if type(condition) is NoAction:
        return SCIPLEX3_CANDIDATE_V5_NO_ACTION_ID
    if type(condition) is CompoundDose:
        return "compound-dose:" + _canonical_json_sha256(
            {"compound": condition.compound, "dose_nm": condition.dose_nm}
        )
    raise SciPlex3CandidateError("candidate condition has an unsupported exact type")


def _v5_target_fingerprint(target: PredictionTarget) -> str:
    condition = target.condition
    condition_payload: dict[str, object]
    if type(condition) is NoAction:
        condition_payload = {"kind": "no_action"}
    elif type(condition) is CompoundDose:
        condition_payload = {
            "compound": condition.compound,
            "dose_nm": condition.dose_nm,
            "kind": "compound_dose",
        }
    else:
        raise SciPlex3CandidateError("candidate target condition has an unsupported exact type")
    return canonical_target_fingerprint(
        {
            "case_id": target.case_id,
            "condition": condition_payload,
            "partition_id": target.partition_id,
            "plate_id": target.plate_id,
            "target_well_id": target.target_well_id,
        }
    )


def _strict_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SciPlex3CandidateError(f"{name} must be a nonblank trimmed string")
    return value


def _strict_sha256(value: object, *, name: str) -> str:
    text = _strict_text(value, name=name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise SciPlex3CandidateError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _require_exact_keys(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SciPlex3CandidateError(
            f"{name} keys differ from the frozen schema; missing={missing}, extra={extra}"
        )


def candidate_specification_manifest() -> dict[str, object]:
    """Return the exact, canonical-JSON-compatible Item 12 scientific specification."""

    return {
        "candidate_specification_schema": SCIPLEX3_CANDIDATE_SPECIFICATION_SCHEMA,
        "candidate_specification_schema_version": (SCIPLEX3_CANDIDATE_SPECIFICATION_SCHEMA_VERSION),
        "implementation_version": SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION,
        "initial_equilibration_manifest": (
            "elbo-order-and-fixed-length-inner-convergence-witnesses"
        ),
        "model_id": SCIPLEX3_CANDIDATE_MODEL_ID,
        "model_schema_version": SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION,
        "context_id": SCIPLEX3_CANDIDATE_CONTEXT_ID,
        "authority": {
            "can_mint_lifecycle_evidence": False,
            "scientifically_admissible": False,
            "exposes_cell_state_belief": False,
            "heldout_memberships_read": False,
            "heldout_outcomes_read": False,
        },
        "support": {
            "feature_count": SCIPLEX3_FEATURE_COUNT,
            "factor_count": SCIPLEX3_CANDIDATE_FACTOR_COUNT,
            "compound_count": SCIPLEX3_CANDIDATE_COMPOUND_COUNT,
            "action_count": SCIPLEX3_CANDIDATE_ACTION_COUNT,
            "doses_nm": list(SCIPLEX3_CANDIDATE_DOSES_NM),
            "no_action_supported": True,
            "target_partitions": list(SCIPLEX3_CANDIDATE_TARGET_PARTITIONS),
            "unseen_compounds_supported": False,
            "dose_interpolation_supported": False,
            "dose_extrapolation_supported": False,
            "maximum_samples_per_request": SCIPLEX3_V5_MAX_SAMPLE_COUNT,
            "request_failure_budget": "2^-64-conditional-int64-tail",
            "sampling_contract_sha256": SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256,
            "supports_argument": "exact-CandidateSampleRequest-not-target-only",
        },
        "distribution": {
            "family": "continuous-gamma-poisson-admixture",
            "categorical_topics": False,
            "loading_prior": {
                "family": "symmetric-dirichlet",
                "concentration": SCIPLEX3_CANDIDATE_DIRICHLET_CONCENTRATION,
                "row_sum": 1.0,
            },
            "capture_multiplier": "fixed-one-no-random-variable",
            "factor_shape": "fixed-r_theta-0.1-not-estimated",
            "fixed_factor_shape": SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE,
            "fixed_factor_shape_hex": SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE.hex(),
            "factors": "theta_k~Gamma(0.1,0.1/m_action_plate_k)",
            "observation": "y_g~Poisson(sum_k(theta_k*B_kg))",
            "zero_inflation": False,
            "hurdle": False,
            "observed_target_depth_conditioning": False,
            "positive_panel_conditioning": ("exact-zero-truncated-compound-poisson-log-series"),
            "positive_panel_rejection_redraws": False,
            "int64_tail_support": "global-action-context-tau-chernoff-request-bound",
        },
        "uncertainty_scope": {
            "included": [
                "poisson-umi-shot-noise",
                "fixed-continuous-gamma-factor-activation-heterogeneity",
            ],
            "excluded_or_unclaimed": [
                "technical-capture-attribution-q-removed",
                "fitted-parameter-or-model-uncertainty",
                "identifiable-action-specific-between-well-variance-one-p1-well-per-action",
                "unseen-plate-context-variation-neutralized",
                "intervention-realization",
                "viability-or-survival",
                "mechanism-pathway-or-cell-state-interpretation",
                "causal-or-transport-uncertainty",
            ],
        },
        "claim_ceiling": {
            "factor_semantics": "statistical-assay-response-factors-only",
            "never_interpret_as": ["cell-types", "pathways", "hidden-cell-state"],
            "predictive_status": "uncalibrated-p1-fit-predictive-association-only",
            "population_scope": "recovered-k562-nuclei",
            "time_scope": "24-hours",
            "support_scope": "exact-panel-actions-and-doses-only",
            "claims_not_made": [
                "causality",
                "intervention-realization",
                "viability-or-survival",
                "mechanism-or-pathway",
                "novel-action-dose-time-system-or-transport",
            ],
        },
        "action_model": {
            "no_action_log_mean": "alpha_k-at-neutral-unseen-plate-context",
            "treated_log_mean": "alpha_k+delta_compound_dose_k-at-neutral-context",
            "canonical_fixed_q_objective": (
                "-(94785/768)*0.1*sum_wk(eta_wk+tbar_wk*exp(-eta_wk))-dose-penalty"
            ),
            "canonical_objective_version": SCIPLEX3_V5_OBJECTIVE_VERSION,
            "equal_well_scale": SCIPLEX3_V5_EQUAL_WELL_SCALE,
            "plate_intercept_fit": "all-768-wells-including-treated-and-vehicle-wells",
            "alpha": "factorwise-logmeanexp-of-eight-fitted-plate-intercepts",
            "rho_normalization": "arithmetic-mean-over-eight-plates-equals-one-per-factor",
            "solver": ("factorwise-arrowhead-newton-with-exact-four-dose-block-schur-complement"),
            "dose_penalty_magnitude_sd": SCIPLEX3_CANDIDATE_DOSE_MAGNITUDE_SD,
            "dose_penalty_second_difference_sd": (SCIPLEX3_CANDIDATE_DOSE_SECOND_DIFFERENCE_SD),
            "monotonicity_constraint": False,
            "terminal_gradient_tolerance": SCIPLEX3_V5_GRADIENT_TOL,
            "accepted_substep_gate": "canonical-fixed-q-full-objective-nondecrease",
            "complete_block_gate": "canonical-and-independent-objectives-agree-and-nondecrease",
        },
        "unseen_plate": {
            "family": "neutral-unit-context",
            "context_count": 1,
            "context_id": SCIPLEX3_CANDIDATE_V5_NEUTRAL_CONTEXT_ID,
            "factor_multiplier": 1.0,
            "deterministic": True,
            "target_plate_context_learned": False,
            "p1_rho_sampling_input": False,
            "rng": RNG_ALGORITHM,
            "factorwise_arithmetic_mean": 1.0,
            "factor_independent_draws": False,
            "parametric_lognormal": False,
            "ordinal_plate_id_embedding": False,
        },
        "fit": {
            "partition_ids": ["p1-train"],
            "replicates": ["rep1"],
            "record_count": SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT,
            "well_count": SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,
            "zero_panel_rows_retained": True,
            "cell_weight": "94785/(768*n_well)",
            "local_posteriors_weighted": False,
            "algorithm": "streamed-sparse-inner-equilibrated-cavi",
            "initialization": "deterministic-nndsvd-canonical-well-means",
            "initialization_score_smoothing": {
                "scope": "initial-nndsvd-well-factor-scores-only",
                "floor_fraction_of_well_total_per_factor": (
                    SCIPLEX3_CANDIDATE_NNDSVD_SCORE_FLOOR_FRACTION
                ),
                "formula": "T*max(s_k,1e-8*T/K)/sum_l(max(s_l,1e-8*T/K))",
                "loading_floor": False,
                "loading_positivity_source": "lambda=.3+1000*B_nndsvd",
                "post_smoothing_score_normalization": "original-well-raw-count-total",
            },
            "batch_size": SCIPLEX3_CANDIDATE_BATCH_SIZE,
            "inner_equilibration": {
                "scope": "row-local-phi-theta-fixed-point-within-canonical-batch",
                "warm_start": "previous-outer-theta-shape",
                "fixed_rate": "0.1/m_well_factor+1",
                "shape_update": "0.1+allocated-counts",
                "shape_residual": (
                    "max(abs(shape_proposal-shape)/max(1,abs(shape),abs(shape_proposal)))"
                ),
                "expected_log_residual": "max(abs(digamma(shape_proposal)-digamma(shape)))",
                "residual_tolerance": SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL,
                "consecutive_passing_sweeps": (SCIPLEX3_CANDIDATE_INNER_CONVERGENCE_STREAK),
                "minimum_sweeps": SCIPLEX3_CANDIDATE_MIN_INNER_SWEEPS,
                "maximum_sweeps": SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS,
                "damping_or_acceleration": False,
                "final_statistics": "second-consecutive-passing-sweep-only",
                "zero_row_shape": SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE,
                "mass_tolerance": "64*eps64*max(1,mass)",
                "global_mass_reconciliation": (
                    "fixed-order-math-fsum-loading-counts-equals-equal-well-weighted-observed-counts"
                ),
                "nonconvergence": "fail-no-candidate",
            },
            "synchronized_schedule": {
                "initial": (
                    "equilibrate-untraced-S0-under-initial-G0-and-store-ELBO-order-inner-witnesses"
                ),
                "outer": ("Gt=M(Q[t-1]);Qt=equilibrate(Gt);Lt=ELBO(Gt,Qt);cache-Qt"),
                "trace_one_relative_reference": "stored-initial-elbo-L0",
                "initial_equilibration_counts_toward_outer_iterations": False,
                "partial_trace_on-block-failure": False,
            },
            "minimum_outer_iterations": SCIPLEX3_CANDIDATE_MIN_OUTER_ITERATIONS,
            "maximum_outer_iterations": SCIPLEX3_CANDIDATE_MAX_OUTER_ITERATIONS,
            "convergence_relative_elbo": SCIPLEX3_CANDIDATE_CONVERGENCE_RTOL,
            "convergence_consecutive_iterations": SCIPLEX3_CANDIDATE_CONVERGENCE_STREAK,
            "factor_shape_mode": "fixed-not-estimated",
            "fixed_factor_shape": SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE,
            "float_precision": "float64",
        },
        "factor_canonicalization": {
            "primary": "descending-equal-well-predictive-mean-contribution",
            "primary_rounding_decimals": SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS,
            "tie_break": "ascending-sha256-rounded-loading-little-endian-float64",
            "duplicate_canonical_key": "fail-degenerate-no-artifact",
            "terminal_order_streak_required": SCIPLEX3_CANDIDATE_CONVERGENCE_STREAK,
        },
        "identifiability": {
            "canonical_rounding_decimals": SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS,
            "mean_activation_witness": (
                "reconstruct-then-round-to-12-decimal-little-endian-float64"
            ),
            "contribution_summation": "math.fsum-canonical-row-order-divide-by-768",
            "reported_rank_ratio_rounding_decimals": (SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS),
            "threshold": SCIPLEX3_CANDIDATE_IDENTIFIABILITY_THRESHOLD,
            "threshold_strict_margin": SCIPLEX3_CANDIDATE_IDENTIFIABILITY_MARGIN,
            "quantization_half_boundary_tolerance": ("64*eps64*max(1,abs(raw_ratio))"),
            "loading_sigma_min_over_max": "strictly-greater-than-threshold",
            "equal_well_activation_sigma_min_over_max": "strictly-greater-than-threshold",
            "minimum_factor_contribution_share": "strictly-greater-than-threshold",
            "contribution_ties": "loading-digest-ordered",
        },
        "sealed_p1_topology": {
            "training_well_ids": "768-unique-canonical-p1-training-order",
            "training_well_plate_indices": "int64[768]",
            "action_well_indices": "int64[188,4]",
            "vehicle_well_indices": "int64[8,2]",
            "action_and_vehicle_partition_all_wells": True,
            "mean_activation": "float64[768,16]-reconstructed-exactly-from-topology-and-means",
            "sampling_input": False,
        },
        "rng_algorithm": RNG_ALGORITHM,
        "rng_substreams": {
            "neutral_context": "single-context-target-key-binding",
            "raw_counts": "model-calibration-target-context-seed-draw-index",
            "prefix_stable_by_draw_index": True,
            "domain_labels_required": True,
        },
        "calibration_declaration_only": {
            "tau_grid": list(SCIPLEX3_CANDIDATE_TAU_GRID),
            "tau_grid_exponents": list(range(-20, 7)),
            "factor_shape_transform": "0.1/tau^2",
            "factor_shape_lower_guard": SCIPLEX3_CANDIDATE_FACTOR_SHAPE_LOWER_GUARD,
            "unseen_plate_context_transform": "unit-context-invariant-under-tau",
            "tau_one_branch": "bit-exact-factor-shape-0.1-and-unit-context",
            "active_training_candidate_tau": 1.0,
            "other_means_weights_and_support_unchanged": True,
            "p2_outcomes_read_by_training": False,
        },
        "reference_runtime": dict(SCIPLEX3_CANDIDATE_REFERENCE_RUNTIME),
    }


SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256: Final = _canonical_json_sha256(
    candidate_specification_manifest()
)


@dataclass(frozen=True, slots=True, order=True)
class SciPlex3P1ActionBinding:
    """One design-authenticated p1 treated well."""

    compound: str
    dose_nm: int
    well_id: str
    plate_id: str

    def __post_init__(self) -> None:
        _strict_text(self.compound, name="action compound")
        _strict_text(self.well_id, name="action well_id")
        _strict_text(self.plate_id, name="action plate_id")
        if type(self.dose_nm) is not int or self.dose_nm not in SCIPLEX3_CANDIDATE_DOSES_NM:
            raise SciPlex3CandidateError("action dose must be one of the four exact frozen doses")

    @property
    def condition(self) -> CompoundDose:
        return CompoundDose(self.compound, self.dose_nm)


@dataclass(frozen=True, slots=True, order=True)
class SciPlex3P1VehicleBinding:
    """The exact pair of p1 vehicle wells on one source plate."""

    plate_id: str
    well_ids: tuple[str, str]

    def __post_init__(self) -> None:
        _strict_text(self.plate_id, name="vehicle plate_id")
        well_ids = tuple(self.well_ids)
        if len(well_ids) != 2 or len(set(well_ids)) != 2:
            raise SciPlex3CandidateError("every p1 plate must bind exactly two vehicle wells")
        for well_id in well_ids:
            _strict_text(well_id, name="vehicle well_id")
        object.__setattr__(self, "well_ids", cast(tuple[str, str], tuple(sorted(well_ids))))


@dataclass(frozen=True, slots=True)
class SciPlex3P1DesignBindings:
    """External design bindings that prevent training metadata from defining its own scope."""

    actions: tuple[SciPlex3P1ActionBinding, ...]
    vehicles: tuple[SciPlex3P1VehicleBinding, ...]
    context_id: str = SCIPLEX3_CANDIDATE_CONTEXT_ID

    def __post_init__(self) -> None:
        if self.context_id != SCIPLEX3_CANDIDATE_CONTEXT_ID:
            raise SciPlex3CandidateError("candidate design context is unsupported")
        actions = tuple(sorted(self.actions))
        vehicles = tuple(sorted(self.vehicles))
        if len(actions) != SCIPLEX3_CANDIDATE_ACTION_COUNT or any(
            type(item) is not SciPlex3P1ActionBinding for item in actions
        ):
            raise SciPlex3CandidateError("candidate design must bind exactly 752 treated actions")
        if len(vehicles) != SCIPLEX3_CANDIDATE_PLATE_COUNT or any(
            type(item) is not SciPlex3P1VehicleBinding for item in vehicles
        ):
            raise SciPlex3CandidateError("candidate design must bind exactly eight p1 plates")
        plate_ids = tuple(item.plate_id for item in vehicles)
        if len(set(plate_ids)) != len(plate_ids):
            raise SciPlex3CandidateError("candidate design contains duplicate plate bindings")
        if any(action.plate_id not in set(plate_ids) for action in actions):
            raise SciPlex3CandidateError("candidate action refers to an unbound p1 plate")
        conditions = tuple((item.compound, item.dose_nm) for item in actions)
        if len(set(conditions)) != len(conditions):
            raise SciPlex3CandidateError("candidate design contains duplicate actions")
        compounds = tuple(sorted({item.compound for item in actions}))
        if len(compounds) != SCIPLEX3_CANDIDATE_COMPOUND_COUNT:
            raise SciPlex3CandidateError("candidate design must contain exactly 188 compounds")
        for compound in compounds:
            doses = tuple(item.dose_nm for item in actions if item.compound == compound)
            if tuple(sorted(doses)) != SCIPLEX3_CANDIDATE_DOSES_NM:
                raise SciPlex3CandidateError(
                    f"candidate compound {compound!r} lacks the four exact frozen doses"
                )
        all_well_ids = [item.well_id for item in actions]
        all_well_ids.extend(well_id for item in vehicles for well_id in item.well_ids)
        if len(set(all_well_ids)) != SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT:
            raise SciPlex3CandidateError("candidate design well IDs are not globally unique")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "vehicles", vehicles)

    @property
    def compounds(self) -> tuple[str, ...]:
        return tuple(sorted({item.compound for item in self.actions}))

    @property
    def plate_ids(self) -> tuple[str, ...]:
        return tuple(item.plate_id for item in self.vehicles)

    def manifest(self) -> dict[str, object]:
        return {
            "context_id": self.context_id,
            "actions": [
                {
                    "compound": item.compound,
                    "dose_nm": item.dose_nm,
                    "plate_id": item.plate_id,
                    "well_id": item.well_id,
                }
                for item in self.actions
            ],
            "vehicles": [
                {"plate_id": item.plate_id, "well_ids": list(item.well_ids)}
                for item in self.vehicles
            ],
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_json_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class SciPlex3CandidateTraceEntry:
    """One immutable outer-loop convergence observation."""

    iteration: int
    elbo: float
    relative_change: float
    factor_order: tuple[int, ...]
    maximum_inner_sweeps: int
    maximum_terminal_shape_residual: float
    maximum_terminal_elog_residual: float
    inner_sweep_count_histogram: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.iteration) is not int
            or not 1 <= self.iteration <= SCIPLEX3_CANDIDATE_MAX_OUTER_ITERATIONS
        ):
            raise SciPlex3CandidateError("trace iteration is outside the frozen outer-loop range")
        if type(self.elbo) is not float or not math.isfinite(self.elbo):
            raise SciPlex3CandidateError("trace ELBO must be a finite float")
        if (
            type(self.relative_change) is not float
            or not math.isfinite(self.relative_change)
            or self.relative_change < 0.0
        ):
            raise SciPlex3CandidateError("trace relative change must be finite and nonnegative")
        order = tuple(self.factor_order)
        if (
            len(order) != SCIPLEX3_CANDIDATE_FACTOR_COUNT
            or any(type(index) is not int for index in order)
            or set(order) != set(range(SCIPLEX3_CANDIDATE_FACTOR_COUNT))
        ):
            raise SciPlex3CandidateError(
                "trace factor order must be an exact permutation of the pre-canonical axes"
            )
        if (
            type(self.maximum_inner_sweeps) is not int
            or not SCIPLEX3_CANDIDATE_MIN_INNER_SWEEPS
            <= self.maximum_inner_sweeps
            <= SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
        ):
            raise SciPlex3CandidateError("trace maximum inner sweeps is invalid")
        for name in (
            "maximum_terminal_shape_residual",
            "maximum_terminal_elog_residual",
        ):
            value = getattr(self, name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not 0.0 <= value <= SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL
            ):
                raise SciPlex3CandidateError(
                    f"trace {name} must be finite and pass the inner tolerance"
                )
        histogram = tuple(self.inner_sweep_count_histogram)
        if (
            len(histogram) != SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
            or any(type(count) is not int or count < 0 for count in histogram)
            or histogram[0] != 0
            or sum(histogram) <= 0
        ):
            raise SciPlex3CandidateError(
                "trace inner-sweep histogram must be a positive fixed-length batch count"
            )
        highest_occupied = max(index + 1 for index, count in enumerate(histogram) if count)
        if highest_occupied != self.maximum_inner_sweeps:
            raise SciPlex3CandidateError(
                "trace maximum inner sweeps differs from its fixed-length histogram"
            )
        object.__setattr__(self, "factor_order", order)
        object.__setattr__(self, "inner_sweep_count_histogram", histogram)


@dataclass(frozen=True, slots=True)
class SciPlex3CandidateInitialEquilibration:
    """Untraced synchronized local optimum under deterministic initialized globals."""

    elbo: float
    factor_order: tuple[int, ...]
    maximum_inner_sweeps: int
    maximum_terminal_shape_residual: float
    maximum_terminal_elog_residual: float
    inner_sweep_count_histogram: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.elbo) is not float or not math.isfinite(self.elbo):
            raise SciPlex3CandidateError("initial-equilibration ELBO must be finite")
        probe = SciPlex3CandidateTraceEntry(
            iteration=1,
            elbo=self.elbo,
            relative_change=0.0,
            factor_order=self.factor_order,
            maximum_inner_sweeps=self.maximum_inner_sweeps,
            maximum_terminal_shape_residual=self.maximum_terminal_shape_residual,
            maximum_terminal_elog_residual=self.maximum_terminal_elog_residual,
            inner_sweep_count_histogram=self.inner_sweep_count_histogram,
        )
        object.__setattr__(self, "factor_order", probe.factor_order)
        object.__setattr__(self, "inner_sweep_count_histogram", probe.inner_sweep_count_histogram)

    def manifest(self) -> dict[str, object]:
        return {
            "elbo": self.elbo,
            "factor_order": list(self.factor_order),
            "inner_sweep_count_histogram": list(self.inner_sweep_count_histogram),
            "maximum_inner_sweeps": self.maximum_inner_sweeps,
            "maximum_terminal_elog_residual": self.maximum_terminal_elog_residual,
            "maximum_terminal_shape_residual": self.maximum_terminal_shape_residual,
        }


@dataclass(frozen=True, slots=True)
class SciPlex3CandidateTrainingSummary:
    """Non-authorizing p1-only facts sealed into model bytes."""

    record_count: int
    well_count: int
    zero_panel_record_count: int
    design_sha256: str
    training_data_sha256: str
    provenance: Literal["real-p1", "synthetic-golden"] = "real-p1"

    def __post_init__(self) -> None:
        for name in ("record_count", "well_count", "zero_panel_record_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SciPlex3CandidateError(f"training summary {name} must be nonnegative")
        if self.record_count <= 0 or self.well_count <= 0:
            raise SciPlex3CandidateError("training summary must contain rows and wells")
        _strict_sha256(self.design_sha256, name="training design_sha256")
        _strict_sha256(self.training_data_sha256, name="training data_sha256")
        if self.provenance not in {"real-p1", "synthetic-golden"}:
            raise SciPlex3CandidateError("training summary provenance is unsupported")


def _freeze_float_array(
    value: object,
    *,
    shape: tuple[int, ...],
    name: str,
    strictly_positive: bool = False,
    nonnegative: bool = False,
) -> FloatArray:
    raw = np.asarray(value, dtype=np.float64, order="C")
    if raw.shape != shape or not bool(np.all(np.isfinite(raw))):
        raise SciPlex3CandidateError(f"{name} must have shape {shape} and finite float64 values")
    if strictly_positive and bool(np.any(raw <= 0.0)):
        raise SciPlex3CandidateError(f"{name} must be strictly positive")
    if nonnegative and bool(np.any(raw < 0.0)):
        raise SciPlex3CandidateError(f"{name} must be nonnegative")
    canonical = np.asarray(raw, dtype="<f8", order="C")
    return np.frombuffer(canonical.tobytes(order="C"), dtype="<f8").reshape(shape)


def _encode_tensor(value: FloatArray) -> dict[str, object]:
    canonical = np.asarray(value, dtype="<f8", order="C")
    data = canonical.tobytes(order="C")
    return {
        "data_base64": base64.b64encode(data).decode("ascii"),
        "dtype": "little-endian-float64",
        "sha256": _sha256_bytes(data),
        "shape": list(canonical.shape),
    }


def _decode_tensor(value: object, *, expected_shape: tuple[int, ...], name: str) -> FloatArray:
    if type(value) is not dict:
        raise SciPlex3CandidateError(f"{name} tensor manifest must be an object")
    manifest = cast(dict[str, object], value)
    _require_exact_keys(manifest, {"data_base64", "dtype", "sha256", "shape"}, name=name)
    if manifest["dtype"] != "little-endian-float64" or manifest["shape"] != list(expected_shape):
        raise SciPlex3CandidateError(f"{name} tensor dtype or shape differs from the frozen schema")
    encoded = _strict_text(manifest["data_base64"], name=f"{name} data_base64")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise SciPlex3CandidateError(f"{name} tensor base64 is invalid") from error
    if base64.b64encode(data).decode("ascii") != encoded:
        raise SciPlex3CandidateError(f"{name} tensor base64 is not canonical")
    expected_byte_count = math.prod(expected_shape) * np.dtype("<f8").itemsize
    if len(data) != expected_byte_count:
        raise SciPlex3CandidateError(f"{name} tensor byte count differs from its shape")
    if _sha256_bytes(data) != _strict_sha256(manifest["sha256"], name=f"{name} sha256"):
        raise SciPlex3CandidateError(f"{name} tensor SHA-256 mismatch")
    array = np.frombuffer(data, dtype="<f8").reshape(expected_shape)
    if not bool(np.all(np.isfinite(array))):
        raise SciPlex3CandidateError(f"{name} tensor contains a nonfinite value")
    return array


def _freeze_int_array(value: object, *, shape: tuple[int, ...], name: str) -> IntArray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in {"i", "u"}:
        raise SciPlex3CandidateError(f"{name} must have shape {shape} and integer values")
    if raw.dtype.kind == "u" and bool(np.any(raw > np.iinfo(np.int64).max)):
        raise SciPlex3CandidateError(f"{name} contains an integer outside signed 64-bit range")
    canonical = np.asarray(raw, dtype="<i8", order="C")
    return np.frombuffer(canonical.tobytes(order="C"), dtype="<i8").reshape(shape)


def _encode_int_tensor(value: IntArray) -> dict[str, object]:
    canonical = np.asarray(value, dtype="<i8", order="C")
    data = canonical.tobytes(order="C")
    return {
        "data_base64": base64.b64encode(data).decode("ascii"),
        "dtype": "little-endian-int64",
        "sha256": _sha256_bytes(data),
        "shape": list(canonical.shape),
    }


def _decode_int_tensor(value: object, *, expected_shape: tuple[int, ...], name: str) -> IntArray:
    if type(value) is not dict:
        raise SciPlex3CandidateError(f"{name} tensor manifest must be an object")
    manifest = cast(dict[str, object], value)
    _require_exact_keys(manifest, {"data_base64", "dtype", "sha256", "shape"}, name=name)
    if manifest["dtype"] != "little-endian-int64" or manifest["shape"] != list(expected_shape):
        raise SciPlex3CandidateError(f"{name} tensor dtype or shape differs from the frozen schema")
    encoded = _strict_text(manifest["data_base64"], name=f"{name} data_base64")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise SciPlex3CandidateError(f"{name} tensor base64 is invalid") from error
    if base64.b64encode(data).decode("ascii") != encoded:
        raise SciPlex3CandidateError(f"{name} tensor base64 is not canonical")
    if len(data) != math.prod(expected_shape) * np.dtype("<i8").itemsize:
        raise SciPlex3CandidateError(f"{name} tensor byte count differs from its shape")
    if _sha256_bytes(data) != _strict_sha256(manifest["sha256"], name=f"{name} sha256"):
        raise SciPlex3CandidateError(f"{name} tensor SHA-256 mismatch")
    return np.frombuffer(data, dtype="<i8").reshape(expected_shape)


def _parse_string_tuple(
    value: object, *, count: int, name: str, require_sorted: bool = True
) -> tuple[str, ...]:
    if type(value) is not list or len(value) != count:
        raise SciPlex3CandidateError(f"{name} must contain exactly {count} strings")
    parsed = tuple(_strict_text(item, name=f"{name} item") for item in value)
    if len(set(parsed)) != len(parsed) or (require_sorted and parsed != tuple(sorted(parsed))):
        ordering = "unique and canonically sorted" if require_sorted else "unique"
        raise SciPlex3CandidateError(f"{name} must be {ordering}")
    return parsed


def _training_summary_manifest(summary: SciPlex3CandidateTrainingSummary) -> dict[str, object]:
    return {
        "design_sha256": summary.design_sha256,
        "partition_ids": ["p1-train"],
        "provenance": summary.provenance,
        "record_count": summary.record_count,
        "replicates": ["rep1"],
        "training_data_sha256": summary.training_data_sha256,
        "well_count": summary.well_count,
        "zero_panel_record_count": summary.zero_panel_record_count,
    }


def _canonical_audit_matrix(value: FloatArray) -> FloatArray:
    rounded = np.asarray(
        np.round(value, SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS),
        dtype="<f8",
        order="C",
    )
    return np.frombuffer(rounded.tobytes(order="C"), dtype="<f8").reshape(rounded.shape)


def _canonical_quantized_scalar(value: float, *, name: str) -> float:
    if not math.isfinite(value):
        raise SciPlex3CandidateError(f"candidate {name} is nonfinite")
    canonical = float(np.round(value, SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS))
    half_quantum = 0.5 * 10.0 ** (-SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS)
    boundary_tolerance = (
        SCIPLEX3_CANDIDATE_QUANTIZATION_BOUNDARY_EPS_MULTIPLIER
        * np.finfo(np.float64).eps
        * max(1.0, abs(value))
    )
    if (
        min(
            abs(value - (canonical - half_quantum)),
            abs(value - (canonical + half_quantum)),
        )
        <= boundary_tolerance
    ):
        raise SciPlex3CandidateError(f"candidate {name} lies on an ambiguous quantization boundary")
    return canonical


def _rounded_matrix_condition_ratio(value: FloatArray, *, name: str) -> float:
    canonical = _canonical_audit_matrix(value)
    singular_values = np.linalg.svd(canonical, compute_uv=False)
    if (
        singular_values.shape != (min(canonical.shape),)
        or not bool(np.all(np.isfinite(singular_values)))
        or singular_values[0] <= 0.0
    ):
        raise SciPlex3CandidateError(f"candidate {name} singular spectrum is invalid")
    ratio = float(singular_values[-1] / singular_values[0])
    canonical_ratio = _canonical_quantized_scalar(ratio, name=f"{name} rank ratio")
    if (
        not math.isfinite(canonical_ratio)
        or canonical_ratio
        <= SCIPLEX3_CANDIDATE_IDENTIFIABILITY_THRESHOLD + SCIPLEX3_CANDIDATE_IDENTIFIABILITY_MARGIN
    ):
        raise SciPlex3CandidateError(
            f"candidate {name} fails the frozen rounded identifiability threshold"
        )
    return canonical_ratio


def _require_declared_tau(tau: object) -> float:
    if type(tau) is not float or tau not in SCIPLEX3_CANDIDATE_TAU_GRID:
        raise SciPlex3CandidateError("candidate tau must be one exact declared p2 grid value")
    return tau


def _factor_shape_for_tau(tau: object) -> float:
    exact_tau = _require_declared_tau(tau)
    value = SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE / (exact_tau * exact_tau)
    if not math.isfinite(value) or value <= SCIPLEX3_CANDIDATE_FACTOR_SHAPE_LOWER_GUARD:
        raise SciPlex3CandidateError("candidate tau violates the retained factor-shape guard")
    return value


def _rho_factorwise_means(rho: FloatArray) -> FloatArray:
    return np.asarray(
        [
            math.fsum(
                float(rho[plate_index, factor_index])
                for plate_index in range(SCIPLEX3_CANDIDATE_PLATE_COUNT)
            )
            / SCIPLEX3_CANDIDATE_PLATE_COUNT
            for factor_index in range(SCIPLEX3_CANDIDATE_FACTOR_COUNT)
        ],
        dtype=np.float64,
    )


def _factor_contribution_shares(contributions: FloatArray) -> FloatArray:
    total = math.fsum(float(value) for value in contributions)
    if not math.isfinite(total) or total <= 0.0:
        raise SciPlex3CandidateError("candidate factor contribution total is invalid")
    shares = np.asarray(contributions / total, dtype=np.float64)
    if (
        not bool(np.all(np.isfinite(shares)))
        or bool(np.any(shares <= SCIPLEX3_CANDIDATE_IDENTIFIABILITY_THRESHOLD))
        or not math.isclose(
            math.fsum(float(value) for value in shares),
            1.0,
            rel_tol=0.0,
            abs_tol=8.0 * np.finfo(np.float64).eps,
        )
    ):
        raise SciPlex3CandidateError(
            "candidate factor contribution shares fail the frozen identifiability gate"
        )
    return shares


def _reconstruct_mean_activation(
    alpha: FloatArray,
    rho: FloatArray,
    delta: FloatArray,
    training_well_plate_indices: IntArray,
    action_well_indices: IntArray,
    vehicle_well_indices: IntArray,
) -> FloatArray:
    means = np.empty(
        (SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT, SCIPLEX3_CANDIDATE_FACTOR_COUNT),
        dtype=np.float64,
    )
    base = np.exp(alpha)[None, :] * rho
    for plate_index in range(SCIPLEX3_CANDIDATE_PLATE_COUNT):
        means[vehicle_well_indices[plate_index]] = base[plate_index]
    for compound_index in range(SCIPLEX3_CANDIDATE_COMPOUND_COUNT):
        for dose_index in range(len(SCIPLEX3_CANDIDATE_DOSES_NM)):
            well_index = int(action_well_indices[compound_index, dose_index])
            plate_index = int(training_well_plate_indices[well_index])
            means[well_index] = base[plate_index] * np.exp(delta[compound_index, dose_index])
    if not bool(np.all(np.isfinite(means))) or bool(np.any(means <= 0.0)):
        raise SciPlex3CandidateError("candidate topology produced invalid mean activations")
    return means


@dataclass(frozen=True, slots=True, eq=False)
class SciPlex3GammaPoissonCandidate:
    """Immutable fitted candidate; this is not a public cell-state estimator."""

    model_id: ClassVar[str] = SCIPLEX3_CANDIDATE_MODEL_ID
    implementation_version: ClassVar[str] = SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION
    converged: ClassVar[bool] = True
    finite: ClassVar[bool] = True

    ordered_feature_keys: tuple[str, ...]
    compounds: tuple[str, ...]
    plate_ids: tuple[str, ...]
    training_well_ids: tuple[str, ...]
    _basis: FloatArray = field(repr=False)
    _alpha: FloatArray = field(repr=False)
    _rho: FloatArray = field(repr=False)
    _delta: FloatArray = field(repr=False)
    _factor_shape: FloatArray = field(repr=False)
    _factor_contributions: FloatArray = field(repr=False)
    _mean_activation: FloatArray = field(repr=False)
    _training_well_plate_indices: IntArray = field(repr=False)
    _action_well_indices: IntArray = field(repr=False)
    _vehicle_well_indices: IntArray = field(repr=False)
    initial_equilibration: SciPlex3CandidateInitialEquilibration
    trace: tuple[SciPlex3CandidateTraceEntry, ...]
    training_summary: SciPlex3CandidateTrainingSummary
    _v5_sampling_parameters_cache: V5SamplingParameters = field(
        init=False, repr=False, compare=False
    )
    _v5_sampling_envelope_certificate_cache: V5SamplingEnvelopeCertificate = field(
        init=False, repr=False, compare=False
    )
    _v5_runtime_sampler_cache: V5PositiveConditionedSampler = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        keys = tuple(self.ordered_feature_keys)
        if len(keys) != SCIPLEX3_FEATURE_COUNT or len(set(keys)) != len(keys):
            raise SciPlex3CandidateError("candidate ordered feature panel must contain 2,000 keys")
        for key in keys:
            _strict_text(key, name="ordered feature key")
        compounds = tuple(self.compounds)
        plates = tuple(self.plate_ids)
        if (
            len(compounds) != SCIPLEX3_CANDIDATE_COMPOUND_COUNT
            or len(set(compounds)) != len(compounds)
            or compounds != tuple(sorted(compounds))
        ):
            raise SciPlex3CandidateError("candidate compounds must be 188 unique sorted labels")
        if (
            len(plates) != SCIPLEX3_CANDIDATE_PLATE_COUNT
            or len(set(plates)) != len(plates)
            or plates != tuple(sorted(plates))
        ):
            raise SciPlex3CandidateError("candidate plates must be eight unique sorted labels")
        for value in (*compounds, *plates):
            _strict_text(value, name="candidate support label")
        training_well_ids = tuple(self.training_well_ids)
        if (
            len(training_well_ids) != SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT
            or len(set(training_well_ids)) != len(training_well_ids)
            or training_well_ids != tuple(sorted(training_well_ids))
        ):
            raise SciPlex3CandidateError(
                "candidate training well IDs must be the 768 unique canonical p1 IDs"
            )
        for well_id in training_well_ids:
            _strict_text(well_id, name="candidate training well ID")
        basis = _freeze_float_array(
            self._basis,
            shape=(SCIPLEX3_CANDIDATE_FACTOR_COUNT, SCIPLEX3_FEATURE_COUNT),
            name="candidate basis",
            strictly_positive=True,
        )
        if not bool(
            np.allclose(
                np.sum(basis, axis=1),
                1.0,
                rtol=0.0,
                atol=5e-13,
            )
        ):
            raise SciPlex3CandidateError("every candidate basis row must sum to one")
        alpha = _freeze_float_array(
            self._alpha, shape=(SCIPLEX3_CANDIDATE_FACTOR_COUNT,), name="candidate alpha"
        )
        rho = _freeze_float_array(
            self._rho,
            shape=(SCIPLEX3_CANDIDATE_PLATE_COUNT, SCIPLEX3_CANDIDATE_FACTOR_COUNT),
            name="candidate rho",
            strictly_positive=True,
        )
        rho_factor_means = _rho_factorwise_means(rho)
        if bool(np.any(rho >= SCIPLEX3_CANDIDATE_PLATE_COUNT)):
            raise SciPlex3CandidateError(
                "candidate rho empirical context multiplier must be strictly less than eight"
            )
        if not bool(np.allclose(rho_factor_means, 1.0, rtol=0.0, atol=5e-13)):
            raise SciPlex3CandidateError(
                "candidate rho must have exact factor-wise arithmetic mean one"
            )
        delta = _freeze_float_array(
            self._delta,
            shape=(
                SCIPLEX3_CANDIDATE_COMPOUND_COUNT,
                len(SCIPLEX3_CANDIDATE_DOSES_NM),
                SCIPLEX3_CANDIDATE_FACTOR_COUNT,
            ),
            name="candidate delta",
        )
        supported_log_means = np.concatenate(
            (alpha[None, :], (delta + alpha[None, None, :]).reshape(-1, alpha.size)),
            axis=0,
        )
        supported_means = np.exp(supported_log_means)
        if not bool(np.all(np.isfinite(supported_means))) or bool(np.any(supported_means <= 0.0)):
            raise SciPlex3CandidateError(
                "candidate supported action means are not finite and strictly positive"
            )
        factor_shape = _freeze_float_array(
            self._factor_shape,
            shape=(1,),
            name="candidate fixed factor shape",
            strictly_positive=True,
        )
        fixed_shape_witness = np.asarray([SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE], dtype="<f8")
        if not np.array_equal(factor_shape, fixed_shape_witness):
            raise SciPlex3CandidateError(
                "candidate factor shape must be the bit-exact fixed 0.1 witness"
            )
        contributions = _freeze_float_array(
            self._factor_contributions,
            shape=(SCIPLEX3_CANDIDATE_FACTOR_COUNT,),
            name="candidate factor contributions",
            strictly_positive=True,
        )
        _factor_contribution_shares(contributions)
        canonical_order = _canonical_factor_order(basis, contributions)
        if not np.array_equal(
            canonical_order,
            np.arange(SCIPLEX3_CANDIDATE_FACTOR_COUNT, dtype=np.int64),
        ):
            raise SciPlex3CandidateError(
                "candidate factors are not in exact canonical contribution/loading order"
            )
        _rounded_matrix_condition_ratio(basis, name="loading matrix")
        training_well_plate_indices = _freeze_int_array(
            self._training_well_plate_indices,
            shape=(SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,),
            name="candidate training-well plate indices",
        )
        if bool(np.any(training_well_plate_indices < 0)) or bool(
            np.any(training_well_plate_indices >= SCIPLEX3_CANDIDATE_PLATE_COUNT)
        ):
            raise SciPlex3CandidateError(
                "candidate training-well plate index is outside p1 plate support"
            )
        action_well_indices = _freeze_int_array(
            self._action_well_indices,
            shape=(SCIPLEX3_CANDIDATE_COMPOUND_COUNT, len(SCIPLEX3_CANDIDATE_DOSES_NM)),
            name="candidate action well indices",
        )
        vehicle_well_indices = _freeze_int_array(
            self._vehicle_well_indices,
            shape=(SCIPLEX3_CANDIDATE_PLATE_COUNT, 2),
            name="candidate vehicle well indices",
        )
        topology = np.concatenate((action_well_indices.ravel(), vehicle_well_indices.ravel()))
        if (
            bool(np.any(topology < 0))
            or bool(np.any(topology >= SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT))
            or not np.array_equal(
                np.sort(topology),
                np.arange(SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT, dtype=np.int64),
            )
        ):
            raise SciPlex3CandidateError(
                "candidate action and vehicle topology must partition all 768 p1 wells"
            )
        for plate_index in range(SCIPLEX3_CANDIDATE_PLATE_COUNT):
            if bool(
                np.any(
                    training_well_plate_indices[vehicle_well_indices[plate_index]] != plate_index
                )
            ):
                raise SciPlex3CandidateError(
                    "candidate vehicle topology differs from its corresponding p1 plate"
                )
        mean_activation = _freeze_float_array(
            self._mean_activation,
            shape=(
                SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,
                SCIPLEX3_CANDIDATE_FACTOR_COUNT,
            ),
            name="candidate mean activation",
            strictly_positive=True,
        )
        reconstructed_activation = _canonical_audit_matrix(
            _reconstruct_mean_activation(
                alpha,
                rho,
                delta,
                training_well_plate_indices,
                action_well_indices,
                vehicle_well_indices,
            )
        )
        if not np.array_equal(mean_activation, reconstructed_activation):
            raise SciPlex3CandidateError(
                "candidate mean activation differs from the sealed p1 topology and parameters"
            )
        expected_contributions = _factor_contributions(mean_activation)
        if not np.array_equal(contributions, expected_contributions):
            raise SciPlex3CandidateError(
                "candidate factor contributions differ from equal-well predictive means"
            )
        _rounded_matrix_condition_ratio(mean_activation, name="mean activation matrix")
        trace = tuple(self.trace)
        if (
            len(trace) < SCIPLEX3_CANDIDATE_MIN_OUTER_ITERATIONS
            or len(trace) > SCIPLEX3_CANDIDATE_MAX_OUTER_ITERATIONS
            or any(type(item) is not SciPlex3CandidateTraceEntry for item in trace)
            or tuple(item.iteration for item in trace) != tuple(range(1, len(trace) + 1))
        ):
            raise SciPlex3CandidateError(
                "candidate convergence trace is incomplete or noncanonical"
            )
        if type(self.training_summary) is not SciPlex3CandidateTrainingSummary:
            raise SciPlex3CandidateError("candidate training summary has the wrong exact type")
        if type(self.initial_equilibration) is not SciPlex3CandidateInitialEquilibration:
            raise SciPlex3CandidateError("candidate initial equilibration has the wrong exact type")
        batch_count = sum(self.initial_equilibration.inner_sweep_count_histogram)
        if any(sum(item.inner_sweep_count_histogram) != batch_count for item in trace):
            raise SciPlex3CandidateError(
                "candidate trace inner-sweep histograms disagree on the p1 batch count"
            )
        previous = self.initial_equilibration.elbo
        for item in trace:
            current = item.elbo
            expected_relative = abs(current - previous) / max(1.0, abs(previous))
            if item.relative_change != expected_relative:
                raise SciPlex3CandidateError("candidate trace contains a forged relative change")
            if current - previous < -SCIPLEX3_CANDIDATE_ELBO_DECREASE_RTOL * max(
                1.0, abs(previous)
            ):
                raise SciPlex3CandidateError("candidate trace contains a material ELBO decrease")
            previous = current
        if any(
            item.relative_change > SCIPLEX3_CANDIDATE_CONVERGENCE_RTOL
            for item in trace[-SCIPLEX3_CANDIDATE_CONVERGENCE_STREAK:]
        ):
            raise SciPlex3CandidateError(
                "candidate trace does not end in the frozen convergence streak"
            )
        terminal = trace[-SCIPLEX3_CANDIDATE_CONVERGENCE_STREAK:]
        if len({item.factor_order for item in terminal}) != 1:
            raise SciPlex3CandidateError(
                "candidate trace does not end in a stable factor-order streak"
            )
        if self.training_summary.provenance == "real-p1":
            if (
                self.training_summary.record_count != SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT
                or self.training_summary.well_count != SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT
                or self.training_summary.zero_panel_record_count
                != SCIPLEX3_CANDIDATE_ZERO_PANEL_RECORD_COUNT
            ):
                raise SciPlex3CandidateError(
                    "real candidate training summary differs from exact p1"
                )
            if _canonical_json_sha256(list(keys)) != SCIPLEX3_ORDERED_FEATURE_KEYS_SHA256:
                raise SciPlex3CandidateError("real candidate feature panel differs from exact p1")
        object.__setattr__(self, "ordered_feature_keys", keys)
        object.__setattr__(self, "compounds", compounds)
        object.__setattr__(self, "plate_ids", plates)
        object.__setattr__(self, "training_well_ids", training_well_ids)
        object.__setattr__(self, "_basis", basis)
        object.__setattr__(self, "_alpha", alpha)
        object.__setattr__(self, "_rho", rho)
        object.__setattr__(self, "_delta", delta)
        object.__setattr__(self, "_factor_shape", factor_shape)
        object.__setattr__(self, "_factor_contributions", contributions)
        object.__setattr__(self, "_mean_activation", mean_activation)
        object.__setattr__(self, "_training_well_plate_indices", training_well_plate_indices)
        object.__setattr__(self, "_action_well_indices", action_well_indices)
        object.__setattr__(self, "_vehicle_well_indices", vehicle_well_indices)
        object.__setattr__(self, "trace", trace)

        # The support envelope is purely numerical, while the row RNG and result provenance bind
        # the final content-addressed model digest.  Cache a placeholder-provenance sampler first
        # so behavior serialization can read its certificate without recursively asking for the
        # model digest; then rebind the same certificate to the completed digest exactly once.
        placeholder_parameters = self._v5_sampling_parameters(model_artifact_sha256="0" * 64)
        placeholder_sampler = V5PositiveConditionedSampler(placeholder_parameters)
        sampling_certificate = placeholder_sampler.envelope_certificate
        expected_combination_count = (
            (SCIPLEX3_CANDIDATE_ACTION_COUNT + 1)
            * len(placeholder_parameters.context_ids)
            * len(SCIPLEX3_CANDIDATE_TAU_GRID)
        )
        if (
            not sampling_certificate.supported
            or sampling_certificate.rejection_reasons
            or sampling_certificate.maximum_request_count != SCIPLEX3_V5_MAX_SAMPLE_COUNT
            or sampling_certificate.combination_count != expected_combination_count
            or sampling_certificate.request_failure_budget_log
            != SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG
            or sampling_certificate.worst_request_tail_log_upper_bound
            > SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG
            or not 0.0
            < sampling_certificate.maximum_compound_poisson_intensity
            <= SCIPLEX3_V5_MAX_COMPOUND_POISSON_INTENSITY
            or sampling_certificate.sampling_contract_sha256 != SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256
            or sampling_certificate.parameter_fingerprint
            != placeholder_parameters.parameter_fingerprint
        ):
            raise SciPlex3CandidateError(
                "candidate numerical state does not support the complete v5 sampling envelope"
            )
        object.__setattr__(self, "_v5_sampling_parameters_cache", placeholder_parameters)
        object.__setattr__(
            self,
            "_v5_sampling_envelope_certificate_cache",
            sampling_certificate,
        )
        object.__setattr__(self, "_v5_runtime_sampler_cache", placeholder_sampler)
        runtime_sampler = placeholder_sampler.with_model_artifact_sha256(self.model_artifact_sha256)
        object.__setattr__(self, "_v5_sampling_parameters_cache", runtime_sampler.parameters)
        object.__setattr__(self, "_v5_runtime_sampler_cache", runtime_sampler)

    @property
    def factor_shape(self) -> float:
        return float(self._factor_shape[0])

    def _target_is_supported(self, target: PredictionTarget) -> bool:
        if target.partition_id not in SCIPLEX3_CANDIDATE_TARGET_PARTITIONS:
            return False
        condition = target.condition
        if type(condition) is NoAction:
            return True
        if (
            type(condition) is not CompoundDose
            or condition.dose_nm not in SCIPLEX3_CANDIDATE_DOSES_NM
        ):
            return False
        return condition.compound in self.compounds

    def _v5_sampling_parameters(self, *, model_artifact_sha256: str) -> V5SamplingParameters:
        action_ids = [SCIPLEX3_CANDIDATE_V5_NO_ACTION_ID]
        action_log_means = [self._alpha]
        for compound_index, compound in enumerate(self.compounds):
            for dose_index, dose_nm in enumerate(SCIPLEX3_CANDIDATE_DOSES_NM):
                action_ids.append(_v5_action_id(CompoundDose(compound, dose_nm)))
                action_log_means.append(self._alpha + self._delta[compound_index, dose_index])
        return V5SamplingParameters(
            model_artifact_sha256=model_artifact_sha256,
            action_ids=tuple(action_ids),
            context_ids=(SCIPLEX3_CANDIDATE_V5_NEUTRAL_CONTEXT_ID,),
            calibration_taus=SCIPLEX3_CANDIDATE_TAU_GRID,
            active_tau=1.0,
            action_log_means=np.stack(action_log_means),
            context_multipliers=np.ones(
                (
                    len(SCIPLEX3_CANDIDATE_TAU_GRID),
                    1,
                    SCIPLEX3_CANDIDATE_FACTOR_COUNT,
                ),
                dtype=np.float64,
            ),
            factor_shapes=np.asarray(
                [_factor_shape_for_tau(tau) for tau in SCIPLEX3_CANDIDATE_TAU_GRID],
                dtype=np.float64,
            ),
            basis=self._basis,
        )

    def _v5_sampling_request(self, request: CandidateSampleRequest) -> V5SampleRequest:
        target = request.target
        return V5SampleRequest(
            target=V5SamplingTarget(
                target_fingerprint=_v5_target_fingerprint(target),
                action_id=_v5_action_id(target.condition),
                context_key=target.plate_id,
            ),
            sample_count=request.sample_count,
            seed=request.seed,
        )

    def _v5_runtime_sampler(self) -> V5PositiveConditionedSampler:
        return self._v5_runtime_sampler_cache

    def supports(self, request: object) -> bool:
        """Return support for one exact count-bearing request; target-only checks fail."""

        if type(request) is not CandidateSampleRequest:
            return False
        exact_request = request
        if not self._target_is_supported(exact_request.target):
            return False
        return self._v5_runtime_sampler().supports(self._v5_sampling_request(exact_request))

    def _sample_validated(
        self,
        request: CandidateSampleRequest,
        *,
        sampler: V5PositiveConditionedSampler | None = None,
        sampling_request: V5SampleRequest | None = None,
    ) -> CandidateRawCountSamples:
        """Execute a structurally validated request through the exact-positive v5 sampler."""

        active_sampler = sampler if sampler is not None else self._v5_runtime_sampler()
        active_request = (
            sampling_request if sampling_request is not None else self._v5_sampling_request(request)
        )
        try:
            sample = active_sampler.sample(active_request)
        except SciPlex3SamplingV5Error as error:
            raise SciPlex3CandidateError("candidate v5 sampling request is unsupported") from error
        return CandidateRawCountSamples(
            candidate_id=self.model_id,
            target=request.target,
            ordered_feature_keys=self.ordered_feature_keys,
            model_artifact_sha256=sample.model_artifact_sha256,
            calibration_state_sha256=sample.calibration_state_sha256,
            sampling_contract_sha256=sample.sampling_contract_sha256,
            target_fingerprint=sample.target_fingerprint,
            context_id=sample.context_id,
            seed=request.seed,
            samples=sample.samples,
        )

    def sample(self, request: object) -> object:
        """Sample exact-positive future raw panels from one fully supported request."""

        if type(request) is not CandidateSampleRequest:
            raise SciPlex3CandidateError(
                "candidate sampling requires an exact CandidateSampleRequest"
            )
        exact_request = request
        if not self._target_is_supported(exact_request.target):
            raise SciPlex3CandidateError(
                "candidate request action, dose, context, count, or RNG support is unsupported"
            )
        sampler = self._v5_runtime_sampler()
        sampling_request = self._v5_sampling_request(exact_request)
        if not sampler.supports(sampling_request):
            raise SciPlex3CandidateError(
                "candidate request action, dose, context, count, or RNG support is unsupported"
            )
        return self._sample_validated(
            exact_request,
            sampler=sampler,
            sampling_request=sampling_request,
        )

    def golden_sample(self) -> CandidateRawCountSamples:
        """Exercise deterministic sampling without naming or reading a protected partition."""

        request = CandidateSampleRequest(
            target=PredictionTarget(
                case_id="candidate-internal-p1-no-action-golden",
                target_well_id="candidate-internal-no-outcome",
                plate_id=self.plate_ids[0],
                partition_id="p1-train",
                condition=NO_ACTION,
            ),
            sample_count=8,
            seed=20260810,
        )
        return self._sample_validated(request)

    def behavior_manifest(self) -> dict[str, object]:
        """Expose exact fail-closed training/runtime gates without granting authority."""

        terminal = self.trace[-SCIPLEX3_CANDIDATE_CONVERGENCE_STREAK:]
        shares = _factor_contribution_shares(self._factor_contributions)
        # These cached fields originate from the placeholder-provenance construction during
        # ``__post_init__``.  Their serialized values are numerical only, so hashing the canonical
        # model remains acyclic; after hashing, the runtime sampler reuses the same certificate with
        # the actual model digest bound to RNG and output provenance.
        sampling_parameters = self._v5_sampling_parameters_cache
        sampling_certificate = self._v5_sampling_envelope_certificate_cache
        return {
            "all_parameters_finite": True,
            "can_mint_lifecycle_evidence": False,
            "capture_latent_present": False,
            "factor_contribution_shares": [float(value) for value in shares],
            "factor_order_stable": len({item.factor_order for item in terminal}) == 1,
            "factor_shape_estimated": False,
            "factor_shape_mode": "fixed",
            "final_elbo": self.trace[-1].elbo,
            "fixed_factor_shape": self.factor_shape,
            "fit_converged": True,
            "heldout_memberships_read": False,
            "heldout_outcomes_read": False,
            "initial_elbo": self.initial_equilibration.elbo,
            "initial_equilibration_sha256": _canonical_json_sha256(
                self.initial_equilibration.manifest()
            ),
            "initial_factor_order": list(self.initial_equilibration.factor_order),
            "initial_inner_sweep_count_histogram": list(
                self.initial_equilibration.inner_sweep_count_histogram
            ),
            "initial_maximum_inner_sweeps": (self.initial_equilibration.maximum_inner_sweeps),
            "initial_maximum_terminal_elog_residual": (
                self.initial_equilibration.maximum_terminal_elog_residual
            ),
            "initial_maximum_terminal_shape_residual": (
                self.initial_equilibration.maximum_terminal_shape_residual
            ),
            "inner_all_batches_converged": True,
            "inner_batch_count": sum(self.trace[0].inner_sweep_count_histogram),
            "inner_equilibration_performed": True,
            "loading_rank_ratio": _rounded_matrix_condition_ratio(
                self._basis, name="loading matrix"
            ),
            "maximum_inner_sweeps": max(item.maximum_inner_sweeps for item in self.trace),
            "maximum_terminal_elog_residual": max(
                item.maximum_terminal_elog_residual for item in self.trace
            ),
            "maximum_terminal_shape_residual": max(
                item.maximum_terminal_shape_residual for item in self.trace
            ),
            "mean_activation_rank_ratio": _rounded_matrix_condition_ratio(
                self._mean_activation, name="mean activation matrix"
            ),
            "minimum_factor_contribution_share": float(np.min(shares)),
            "model_schema_version": SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION,
            "outer_iteration_count": len(self.trace),
            "plate_context_count": 1,
            "plate_context_factorwise_mean_one": True,
            "plate_context_family": "neutral-unit-context",
            "sampling_active_calibration_state_sha256": (
                sampling_parameters.active_calibration_state_sha256
            ),
            "sampling_contract_sha256": SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256,
            "sampling_envelope_combination_count": sampling_certificate.combination_count,
            "sampling_envelope_maximum_compound_poisson_intensity": (
                sampling_certificate.maximum_compound_poisson_intensity
            ),
            "sampling_envelope_maximum_request_count": (sampling_certificate.maximum_request_count),
            "sampling_envelope_rejection_reasons": list(sampling_certificate.rejection_reasons),
            "sampling_envelope_request_failure_budget_log": (
                sampling_certificate.request_failure_budget_log
            ),
            "sampling_envelope_supported": sampling_certificate.supported,
            "sampling_envelope_worst_request_tail_log_upper_bound": (
                sampling_certificate.worst_request_tail_log_upper_bound
            ),
            "scientifically_admissible": False,
            "terminal_elbo_relative_changes": [item.relative_change for item in terminal],
            "training_partition_ids": ["p1-train"],
        }

    def _tensor_manifests(self) -> dict[str, object]:
        return {
            "action_well_indices": _encode_int_tensor(self._action_well_indices),
            "alpha": _encode_tensor(self._alpha),
            "basis": _encode_tensor(self._basis),
            "delta": _encode_tensor(self._delta),
            "factor_contributions": _encode_tensor(self._factor_contributions),
            "factor_shape": _encode_tensor(self._factor_shape),
            "mean_activation": _encode_tensor(self._mean_activation),
            "rho": _encode_tensor(self._rho),
            "training_well_plate_indices": _encode_int_tensor(self._training_well_plate_indices),
            "vehicle_well_indices": _encode_int_tensor(self._vehicle_well_indices),
        }

    def fitted_state_manifest(self) -> dict[str, object]:
        """Return exact state identities, never protected outcomes or lifecycle evidence."""

        tensor_manifests = self._tensor_manifests()
        tensor_hashes = {
            name: cast(dict[str, object], manifest)["sha256"]
            for name, manifest in tensor_manifests.items()
        }
        return {
            "behavior": self.behavior_manifest(),
            "candidate_specification_sha256": SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256,
            "compounds_sha256": _canonical_json_sha256(list(self.compounds)),
            "implementation_version": self.implementation_version,
            "initial_equilibration_sha256": _canonical_json_sha256(
                self.initial_equilibration.manifest()
            ),
            "inner_equilibration_trace_sha256": _canonical_json_sha256(
                [
                    self.initial_equilibration.manifest(),
                    *[
                        {
                            "inner_sweep_count_histogram": list(item.inner_sweep_count_histogram),
                            "iteration": item.iteration,
                            "maximum_inner_sweeps": item.maximum_inner_sweeps,
                            "maximum_terminal_elog_residual": (item.maximum_terminal_elog_residual),
                            "maximum_terminal_shape_residual": (
                                item.maximum_terminal_shape_residual
                            ),
                        }
                        for item in self.trace
                    ],
                ]
            ),
            "model_id": self.model_id,
            "model_schema_version": SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION,
            "ordered_feature_keys_sha256": _canonical_json_sha256(list(self.ordered_feature_keys)),
            "plate_ids_sha256": _canonical_json_sha256(list(self.plate_ids)),
            "tensor_sha256": tensor_hashes,
            "training": _training_summary_manifest(self.training_summary),
            "training_well_ids_sha256": _canonical_json_sha256(list(self.training_well_ids)),
        }

    def _payload(self) -> dict[str, object]:
        return {
            "authority": {
                "can_mint_lifecycle_evidence": False,
                "exposes_cell_state_belief": False,
                "scientifically_admissible": False,
            },
            "behavior": self.behavior_manifest(),
            "candidate_specification": candidate_specification_manifest(),
            "compounds": list(self.compounds),
            "implementation_version": self.implementation_version,
            "initial_equilibration": self.initial_equilibration.manifest(),
            "model_id": self.model_id,
            "model_schema": SCIPLEX3_CANDIDATE_MODEL_SCHEMA,
            "model_schema_version": SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION,
            "ordered_feature_keys": list(self.ordered_feature_keys),
            "plate_ids": list(self.plate_ids),
            "tensors": self._tensor_manifests(),
            "trace": [
                {
                    "elbo": item.elbo,
                    "factor_order": list(item.factor_order),
                    "inner_sweep_count_histogram": list(item.inner_sweep_count_histogram),
                    "iteration": item.iteration,
                    "maximum_inner_sweeps": item.maximum_inner_sweeps,
                    "maximum_terminal_elog_residual": (item.maximum_terminal_elog_residual),
                    "maximum_terminal_shape_residual": (item.maximum_terminal_shape_residual),
                    "relative_change": item.relative_change,
                }
                for item in self.trace
            ],
            "training": _training_summary_manifest(self.training_summary),
            "training_well_ids": list(self.training_well_ids),
        }

    def canonical_model_bytes(self) -> bytes:
        """Return the sealed full model as canonical JSON bytes."""

        return _canonical_json_bytes(self._payload())

    def model_bytes(self) -> bytes:
        """Alias used by the trusted candidate-factory interface."""

        return self.canonical_model_bytes()

    @property
    def model_artifact_sha256(self) -> str:
        """Content identity of the complete canonical model bytes."""

        return _sha256_bytes(self.canonical_model_bytes())

    @classmethod
    def fit(
        cls, training: P1TrainingData, design: SciPlex3P1DesignBindings
    ) -> SciPlex3GammaPoissonCandidate:
        """Fit the exact deterministic p1-only candidate or fail without emitting an artifact."""

        candidate = _fit_sciplex3_candidate_exact(training, design)
        if type(candidate) is not cls:
            raise SciPlex3CandidateError("candidate fitter returned a substituted implementation")
        return candidate

    @classmethod
    def from_canonical_model_bytes(cls, payload: bytes) -> SciPlex3GammaPoissonCandidate:
        """Load only exact canonical bytes with complete schema and tensor validation."""

        if type(payload) is not bytes:
            raise SciPlex3CandidateError("candidate model payload must be exact immutable bytes")
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SciPlex3CandidateError(
                "candidate model payload is not valid UTF-8 JSON"
            ) from error
        if type(decoded) is not dict or _canonical_json_bytes(decoded) != payload:
            raise SciPlex3CandidateError("candidate model payload is not exact canonical JSON")
        root = cast(dict[str, object], decoded)
        _require_exact_keys(
            root,
            {
                "authority",
                "behavior",
                "candidate_specification",
                "compounds",
                "implementation_version",
                "initial_equilibration",
                "model_id",
                "model_schema",
                "model_schema_version",
                "ordered_feature_keys",
                "plate_ids",
                "tensors",
                "trace",
                "training",
                "training_well_ids",
            },
            name="candidate model",
        )
        if (
            root["model_schema"] != SCIPLEX3_CANDIDATE_MODEL_SCHEMA
            or root["model_schema_version"] != SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION
            or root["model_id"] != SCIPLEX3_CANDIDATE_MODEL_ID
            or root["implementation_version"] != SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION
            or root["candidate_specification"] != candidate_specification_manifest()
        ):
            raise SciPlex3CandidateError("candidate identity or scientific specification drifted")
        expected_authority = {
            "can_mint_lifecycle_evidence": False,
            "exposes_cell_state_belief": False,
            "scientifically_admissible": False,
        }
        if root["authority"] != expected_authority:
            raise SciPlex3CandidateError("candidate authority boundary drifted")
        keys = _parse_string_tuple(
            root["ordered_feature_keys"],
            count=SCIPLEX3_FEATURE_COUNT,
            name="feature keys",
            require_sorted=False,
        )
        compounds = _parse_string_tuple(
            root["compounds"], count=SCIPLEX3_CANDIDATE_COMPOUND_COUNT, name="compounds"
        )
        plates = _parse_string_tuple(
            root["plate_ids"], count=SCIPLEX3_CANDIDATE_PLATE_COUNT, name="plate IDs"
        )
        training_well_ids = _parse_string_tuple(
            root["training_well_ids"],
            count=SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,
            name="training well IDs",
        )
        tensors_raw = root["tensors"]
        if type(tensors_raw) is not dict:
            raise SciPlex3CandidateError("candidate tensors must be an object")
        tensors = cast(dict[str, object], tensors_raw)
        _require_exact_keys(
            tensors,
            {
                "action_well_indices",
                "alpha",
                "basis",
                "delta",
                "factor_contributions",
                "factor_shape",
                "mean_activation",
                "rho",
                "training_well_plate_indices",
                "vehicle_well_indices",
            },
            name="candidate tensors",
        )
        trace_raw = root["trace"]
        if type(trace_raw) is not list:
            raise SciPlex3CandidateError("candidate trace must be an array")
        initial_raw = root["initial_equilibration"]
        if type(initial_raw) is not dict:
            raise SciPlex3CandidateError("candidate initial equilibration must be an object")
        initial = cast(dict[str, object], initial_raw)
        _require_exact_keys(
            initial,
            {
                "elbo",
                "factor_order",
                "inner_sweep_count_histogram",
                "maximum_inner_sweeps",
                "maximum_terminal_elog_residual",
                "maximum_terminal_shape_residual",
            },
            name="initial equilibration",
        )
        if (
            type(initial["elbo"]) is not float
            or type(initial["factor_order"]) is not list
            or type(initial["inner_sweep_count_histogram"]) is not list
            or type(initial["maximum_inner_sweeps"]) is not int
            or type(initial["maximum_terminal_elog_residual"]) is not float
            or type(initial["maximum_terminal_shape_residual"]) is not float
        ):
            raise SciPlex3CandidateError("candidate initial-equilibration types are invalid")
        initial_order_raw = cast(list[object], initial["factor_order"])
        initial_histogram_raw = cast(list[object], initial["inner_sweep_count_histogram"])
        if any(type(value) is not int for value in initial_order_raw) or any(
            type(value) is not int for value in initial_histogram_raw
        ):
            raise SciPlex3CandidateError(
                "candidate initial-equilibration arrays have invalid scalar types"
            )
        initial_equilibration = SciPlex3CandidateInitialEquilibration(
            elbo=initial["elbo"],
            factor_order=tuple(cast(list[int], initial_order_raw)),
            maximum_inner_sweeps=initial["maximum_inner_sweeps"],
            maximum_terminal_shape_residual=initial["maximum_terminal_shape_residual"],
            maximum_terminal_elog_residual=initial["maximum_terminal_elog_residual"],
            inner_sweep_count_histogram=tuple(cast(list[int], initial_histogram_raw)),
        )
        trace: list[SciPlex3CandidateTraceEntry] = []
        for raw_entry in trace_raw:
            if type(raw_entry) is not dict:
                raise SciPlex3CandidateError("candidate trace entry must be an object")
            entry = cast(dict[str, object], raw_entry)
            _require_exact_keys(
                entry,
                {
                    "elbo",
                    "factor_order",
                    "inner_sweep_count_histogram",
                    "iteration",
                    "maximum_inner_sweeps",
                    "maximum_terminal_elog_residual",
                    "maximum_terminal_shape_residual",
                    "relative_change",
                },
                name="trace entry",
            )
            if (
                type(entry["iteration"]) is not int
                or type(entry["elbo"]) is not float
                or type(entry["factor_order"]) is not list
                or type(entry["maximum_inner_sweeps"]) is not int
                or type(entry["maximum_terminal_elog_residual"]) is not float
                or type(entry["maximum_terminal_shape_residual"]) is not float
                or type(entry["inner_sweep_count_histogram"]) is not list
            ):
                raise SciPlex3CandidateError("candidate trace scalar types are invalid")
            relative = entry["relative_change"]
            if type(relative) is not float:
                raise SciPlex3CandidateError("candidate trace relative-change type is invalid")
            factor_order_raw = cast(list[object], entry["factor_order"])
            if any(type(value) is not int for value in factor_order_raw):
                raise SciPlex3CandidateError("candidate trace factor-order type is invalid")
            histogram_raw = cast(list[object], entry["inner_sweep_count_histogram"])
            if any(type(value) is not int for value in histogram_raw):
                raise SciPlex3CandidateError("candidate trace inner-histogram type is invalid")
            trace.append(
                SciPlex3CandidateTraceEntry(
                    iteration=entry["iteration"],
                    elbo=entry["elbo"],
                    relative_change=relative,
                    factor_order=tuple(cast(list[int], factor_order_raw)),
                    maximum_inner_sweeps=entry["maximum_inner_sweeps"],
                    maximum_terminal_shape_residual=entry["maximum_terminal_shape_residual"],
                    maximum_terminal_elog_residual=entry["maximum_terminal_elog_residual"],
                    inner_sweep_count_histogram=tuple(cast(list[int], histogram_raw)),
                )
            )
        training_raw = root["training"]
        if type(training_raw) is not dict:
            raise SciPlex3CandidateError("candidate training summary must be an object")
        training = cast(dict[str, object], training_raw)
        _require_exact_keys(
            training,
            {
                "design_sha256",
                "partition_ids",
                "provenance",
                "record_count",
                "replicates",
                "training_data_sha256",
                "well_count",
                "zero_panel_record_count",
            },
            name="candidate training summary",
        )
        if training["partition_ids"] != ["p1-train"] or training["replicates"] != ["rep1"]:
            raise SciPlex3CandidateError("candidate training scope is not exact p1 replicate 1")
        if (
            any(
                type(training[name]) is not int
                for name in ("record_count", "well_count", "zero_panel_record_count")
            )
            or type(training["provenance"]) is not str
        ):
            raise SciPlex3CandidateError("candidate training summary scalar types are invalid")
        summary = SciPlex3CandidateTrainingSummary(
            record_count=cast(int, training["record_count"]),
            well_count=cast(int, training["well_count"]),
            zero_panel_record_count=cast(int, training["zero_panel_record_count"]),
            design_sha256=cast(str, training["design_sha256"]),
            training_data_sha256=cast(str, training["training_data_sha256"]),
            provenance=cast(Literal["real-p1", "synthetic-golden"], training["provenance"]),
        )
        candidate = cls(
            ordered_feature_keys=keys,
            compounds=compounds,
            plate_ids=plates,
            training_well_ids=training_well_ids,
            _basis=_decode_tensor(
                tensors["basis"],
                expected_shape=(SCIPLEX3_CANDIDATE_FACTOR_COUNT, SCIPLEX3_FEATURE_COUNT),
                name="basis",
            ),
            _alpha=_decode_tensor(
                tensors["alpha"],
                expected_shape=(SCIPLEX3_CANDIDATE_FACTOR_COUNT,),
                name="alpha",
            ),
            _rho=_decode_tensor(
                tensors["rho"],
                expected_shape=(
                    SCIPLEX3_CANDIDATE_PLATE_COUNT,
                    SCIPLEX3_CANDIDATE_FACTOR_COUNT,
                ),
                name="rho",
            ),
            _delta=_decode_tensor(
                tensors["delta"],
                expected_shape=(
                    SCIPLEX3_CANDIDATE_COMPOUND_COUNT,
                    len(SCIPLEX3_CANDIDATE_DOSES_NM),
                    SCIPLEX3_CANDIDATE_FACTOR_COUNT,
                ),
                name="delta",
            ),
            _factor_shape=_decode_tensor(
                tensors["factor_shape"],
                expected_shape=(1,),
                name="factor_shape",
            ),
            _factor_contributions=_decode_tensor(
                tensors["factor_contributions"],
                expected_shape=(SCIPLEX3_CANDIDATE_FACTOR_COUNT,),
                name="factor_contributions",
            ),
            _mean_activation=_decode_tensor(
                tensors["mean_activation"],
                expected_shape=(
                    SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,
                    SCIPLEX3_CANDIDATE_FACTOR_COUNT,
                ),
                name="mean_activation",
            ),
            _training_well_plate_indices=_decode_int_tensor(
                tensors["training_well_plate_indices"],
                expected_shape=(SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,),
                name="training_well_plate_indices",
            ),
            _action_well_indices=_decode_int_tensor(
                tensors["action_well_indices"],
                expected_shape=(
                    SCIPLEX3_CANDIDATE_COMPOUND_COUNT,
                    len(SCIPLEX3_CANDIDATE_DOSES_NM),
                ),
                name="action_well_indices",
            ),
            _vehicle_well_indices=_decode_int_tensor(
                tensors["vehicle_well_indices"],
                expected_shape=(SCIPLEX3_CANDIDATE_PLATE_COUNT, 2),
                name="vehicle_well_indices",
            ),
            initial_equilibration=initial_equilibration,
            trace=tuple(trace),
            training_summary=summary,
        )
        if root["behavior"] != candidate.behavior_manifest():
            raise SciPlex3CandidateError("candidate behavior gates disagree with fitted state")
        return candidate

    @classmethod
    def load_exact(
        cls, model_bytes: bytes, *, expected_sha256: str
    ) -> SciPlex3GammaPoissonCandidate:
        """Authenticate external content identity before parsing any model field."""

        expected = _strict_sha256(expected_sha256, name="expected model sha256")
        if type(model_bytes) is not bytes or _sha256_bytes(model_bytes) != expected:
            raise SciPlex3CandidateError(
                "candidate model bytes differ from externally bound SHA-256"
            )
        return cls.from_canonical_model_bytes(model_bytes)


def candidate_model_schema_manifest() -> dict[str, object]:
    """Describe the exact sealed output type and its trusted loading surface."""

    return {
        "canonical_encoding": "utf8-canonical-json-sort-keys-no-whitespace-no-nan",
        "canonical_audit_matrix": "round-12-little-endian-float64-bytes-reload",
        "candidate_class": (
            "cellstate.evaluation.sciplex3_candidate.SciPlex3GammaPoissonCandidate"
        ),
        "implementation_version": SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION,
        "initial_equilibration_manifest": (
            "elbo-order-and-fixed-length-inner-convergence-witnesses"
        ),
        "load_entrypoint": "load_sciplex3_candidate",
        "load_exact_requires_external_sha256": True,
        "model_id": SCIPLEX3_CANDIDATE_MODEL_ID,
        "model_schema": SCIPLEX3_CANDIDATE_MODEL_SCHEMA,
        "model_schema_version": SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION,
        "model_bytes_entrypoint": "canonical_model_bytes",
        "output_model_topology": SCIPLEX3_CANDIDATE_OUTPUT_MODEL_TOPOLOGY,
        "request_class": "cellstate.evaluation.sciplex3_candidate.CandidateSampleRequest",
        "result_class": "cellstate.evaluation.sciplex3_candidate.CandidateRawCountSamples",
        "sampling_contract_sha256": SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256,
        "tensor_encoding": "base64-canonical-little-endian-with-sha256",
        "derived_witnesses": {
            "factor_contributions": "fixed-row-order-math-fsum-of-mean_activation/768",
            "factor_shape": "bit-exact-fixed-0.1-not-estimated",
            "mean_activation": "canonical-audit-matrix-reconstructed-from-p1-topology-and-means",
            "rho": "eight-positive-whole-plate-context-rows-factorwise-arithmetic-mean-one",
        },
        "plate_context_family": "neutral-unit-context",
        "required_model_keys": [
            "authority",
            "behavior",
            "candidate_specification",
            "compounds",
            "implementation_version",
            "initial_equilibration",
            "model_id",
            "model_schema",
            "model_schema_version",
            "ordered_feature_keys",
            "plate_ids",
            "tensors",
            "trace",
            "training",
            "training_well_ids",
        ],
        "tensor_names": [
            "action_well_indices",
            "alpha",
            "basis",
            "delta",
            "factor_contributions",
            "factor_shape",
            "mean_activation",
            "rho",
            "training_well_plate_indices",
            "vehicle_well_indices",
        ],
        "training_well_ids_count": SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,
    }


SCIPLEX3_CANDIDATE_OUTPUT_MODEL_SCHEMA_SHA256: Final = _canonical_json_sha256(
    candidate_model_schema_manifest()
)


def _array_identity(value: NDArray[np.generic], *, dtype: str) -> dict[str, object]:
    canonical = np.asarray(value, dtype=dtype, order="C")
    data = canonical.tobytes(order="C")
    return {"sha256": _sha256_bytes(data), "shape": list(canonical.shape)}


def training_data_fingerprint(training: P1TrainingData) -> str:
    """Hash the complete immutable p1 training surface without materializing dense rows."""

    if type(training) is not P1TrainingData:
        raise SciPlex3CandidateError("candidate fitting requires exact P1TrainingData")
    wells: list[dict[str, object]] = []
    for well in training.wells:
        condition: dict[str, object]
        if well.condition is None:
            condition = {"active_dose_nm": 0, "kind": "no_action"}
        else:
            condition = {
                "compound": well.condition.compound,
                "dose_nm": well.condition.dose_nm,
                "kind": "compound_dose",
            }
        wells.append(
            {
                "condition": condition,
                "counts": {
                    "feature_indices": _array_identity(well.counts.feature_indices, dtype="<i8"),
                    "indptr": _array_identity(well.counts.indptr, dtype="<i8"),
                    "values": _array_identity(well.counts.values, dtype="<i8"),
                },
                "partition_id": well.partition_id,
                "plate_id": well.plate_id,
                "record_ids_sha256": _canonical_json_sha256(list(well.record_ids)),
                "replicate": well.replicate,
                "source_row_indices_sha256": _canonical_json_sha256(list(well.source_row_indices)),
                "well_id": well.well_id,
            }
        )
    return _canonical_json_sha256(
        {
            "fingerprint_schema": "sciplex3-candidate-p1-training-data-v1",
            "ordered_feature_keys_sha256": _canonical_json_sha256(
                list(training.ordered_feature_keys)
            ),
            "wells": wells,
        }
    )


@dataclass(frozen=True, slots=True)
class _ValidatedTrainingDesign:
    wells: tuple[P1WellCounts, ...]
    well_index_by_id: Mapping[str, int]
    action_well_indices: IntArray
    action_plate_indices: IntArray
    vehicle_well_indices: IntArray
    training_well_plate_indices: IntArray
    record_count: int
    zero_panel_record_count: int


def _validate_training_design(
    training: P1TrainingData, design: SciPlex3P1DesignBindings
) -> _ValidatedTrainingDesign:
    if type(training) is not P1TrainingData:
        raise SciPlex3CandidateError("candidate fitting requires exact P1TrainingData")
    if type(design) is not SciPlex3P1DesignBindings:
        raise SciPlex3CandidateError("candidate fitting requires exact external p1 design bindings")
    wells = training.wells
    if len(wells) != SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT:
        raise SciPlex3CandidateError("candidate fitting requires exactly 768 p1 wells")
    record_count = sum(well.counts.row_count for well in wells)
    if record_count != SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT:
        raise SciPlex3CandidateError("candidate fitting requires exactly 94,785 p1 records")
    zero_count = sum(
        int(np.sum(np.diff(well.counts.indptr) == 0, dtype=np.int64)) for well in wells
    )
    if zero_count != SCIPLEX3_CANDIDATE_ZERO_PANEL_RECORD_COUNT:
        raise SciPlex3CandidateError("candidate fitting must retain the seven zero-panel p1 rows")
    index_by_id = {well.well_id: index for index, well in enumerate(wells)}
    if len(index_by_id) != len(wells):
        raise SciPlex3CandidateError("candidate p1 well IDs are not unique")
    action_indices = np.empty(
        (SCIPLEX3_CANDIDATE_COMPOUND_COUNT, len(SCIPLEX3_CANDIDATE_DOSES_NM)),
        dtype=np.int64,
    )
    action_plate_indices = np.empty_like(action_indices)
    compound_index = {compound: index for index, compound in enumerate(design.compounds)}
    plate_index = {plate: index for index, plate in enumerate(design.plate_ids)}
    try:
        training_well_plate_indices = np.asarray(
            [plate_index[well.plate_id] for well in wells], dtype=np.int64
        )
    except KeyError as error:
        raise SciPlex3CandidateError(
            "p1 training well refers to a plate outside the external design"
        ) from error
    for action_binding in design.actions:
        try:
            well_index = index_by_id[action_binding.well_id]
        except KeyError as error:
            raise SciPlex3CandidateError(
                "design-bound treated well is absent from p1 data"
            ) from error
        well = wells[well_index]
        if well.condition != action_binding.condition or well.plate_id != action_binding.plate_id:
            raise SciPlex3CandidateError("p1 treated well metadata differs from external design")
        c_index = compound_index[action_binding.compound]
        d_index = SCIPLEX3_CANDIDATE_DOSES_NM.index(action_binding.dose_nm)
        action_indices[c_index, d_index] = well_index
        action_plate_indices[c_index, d_index] = plate_index[action_binding.plate_id]
    vehicle_indices = np.empty(
        (SCIPLEX3_CANDIDATE_PLATE_COUNT, 2),
        dtype=np.int64,
    )
    for vehicle_binding in design.vehicles:
        p_index = plate_index[vehicle_binding.plate_id]
        for control_index, well_id in enumerate(vehicle_binding.well_ids):
            try:
                well_index = index_by_id[well_id]
            except KeyError as error:
                raise SciPlex3CandidateError(
                    "design-bound vehicle well is absent from p1 data"
                ) from error
            well = wells[well_index]
            if not well.is_vehicle or well.plate_id != vehicle_binding.plate_id:
                raise SciPlex3CandidateError(
                    "p1 vehicle well metadata differs from external design"
                )
            vehicle_indices[p_index, control_index] = well_index
    bound_indices = set(action_indices.ravel().tolist()) | set(vehicle_indices.ravel().tolist())
    if bound_indices != set(range(len(wells))):
        raise SciPlex3CandidateError("external design does not exhaust the exact p1 well set")
    return _ValidatedTrainingDesign(
        wells=wells,
        well_index_by_id=MappingProxyType(index_by_id),
        action_well_indices=_freeze_int_array(
            action_indices, shape=action_indices.shape, name="action well indices"
        ),
        action_plate_indices=_freeze_int_array(
            action_plate_indices,
            shape=action_plate_indices.shape,
            name="action plate indices",
        ),
        vehicle_well_indices=_freeze_int_array(
            vehicle_indices, shape=vehicle_indices.shape, name="vehicle well indices"
        ),
        training_well_plate_indices=_freeze_int_array(
            training_well_plate_indices,
            shape=(SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,),
            name="training well plate indices",
        ),
        record_count=record_count,
        zero_panel_record_count=zero_count,
    )


def _nndsvd_initialization(well_means: FloatArray) -> tuple[FloatArray, FloatArray]:
    if well_means.shape != (SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT, SCIPLEX3_FEATURE_COUNT):
        raise SciPlex3CandidateError("NNDSVD input differs from canonical 768x2000 well means")
    if not bool(np.all(np.isfinite(well_means))) or bool(np.any(well_means < 0.0)):
        raise SciPlex3CandidateError("NNDSVD input must be finite nonnegative raw-count means")
    start = np.linspace(1.0, 2.0, min(well_means.shape), dtype=np.float64)
    start /= np.linalg.norm(start)
    try:
        left, singular_values, right = svds(
            well_means,
            k=SCIPLEX3_CANDIDATE_FACTOR_COUNT + 1,
            which="LM",
            solver="arpack",
            v0=start,
            return_singular_vectors=True,
        )
    except Exception as error:  # scipy uses solver-specific exception classes
        raise SciPlex3CandidateError("deterministic NNDSVD singular-value solve failed") from error
    order = np.argsort(singular_values)[::-1]
    singular_values = singular_values[order]
    left = left[:, order]
    right = right[order]
    if not bool(np.all(np.isfinite(singular_values))) or singular_values[15] <= 0.0:
        raise SciPlex3CandidateError("NNDSVD retained singular spectrum is degenerate")
    if bool(
        np.isclose(
            singular_values[15],
            singular_values[16],
            rtol=SCIPLEX3_CANDIDATE_SVD_BOUNDARY_TIE_RTOL,
            atol=0.0,
        )
    ):
        raise SciPlex3CandidateError("NNDSVD has an S16/S17 retained-boundary tie")
    rank = SCIPLEX3_CANDIDATE_FACTOR_COUNT
    scores = np.zeros((well_means.shape[0], rank), dtype=np.float64)
    loadings = np.zeros((rank, well_means.shape[1]), dtype=np.float64)
    scores[:, 0] = math.sqrt(float(singular_values[0])) * np.abs(left[:, 0])
    loadings[0] = math.sqrt(float(singular_values[0])) * np.abs(right[0])
    for component in range(1, rank):
        left_vector = left[:, component]
        right_vector = right[component]
        left_positive = np.maximum(left_vector, 0.0)
        left_negative = np.maximum(-left_vector, 0.0)
        right_positive = np.maximum(right_vector, 0.0)
        right_negative = np.maximum(-right_vector, 0.0)
        positive_norm = np.linalg.norm(left_positive) * np.linalg.norm(right_positive)
        negative_norm = np.linalg.norm(left_negative) * np.linalg.norm(right_negative)
        if positive_norm >= negative_norm:
            chosen_left, chosen_right, product_norm = (
                left_positive,
                right_positive,
                positive_norm,
            )
        else:
            chosen_left, chosen_right, product_norm = (
                left_negative,
                right_negative,
                negative_norm,
            )
        if product_norm <= np.finfo(np.float64).tiny:
            raise SciPlex3CandidateError("NNDSVD produced a degenerate retained factor")
        scale = math.sqrt(float(singular_values[component]) * product_norm)
        scores[:, component] = scale * chosen_left / np.linalg.norm(chosen_left)
        loadings[component] = scale * chosen_right / np.linalg.norm(chosen_right)
    loading_totals = np.sum(loadings, axis=1)
    if bool(np.any(loading_totals <= 0.0)):
        raise SciPlex3CandidateError("NNDSVD loading normalization is degenerate")
    basis = loadings / loading_totals[:, None]
    scores *= loading_totals[None, :]
    row_totals = np.sum(well_means, axis=1)
    score_totals = np.sum(scores, axis=1)
    if bool(np.any(row_totals <= 0.0)) or bool(np.any(score_totals <= 0.0)):
        raise SciPlex3CandidateError("NNDSVD well-score total is degenerate")
    scores *= (row_totals / score_totals)[:, None]
    return _regularize_nndsvd_initialization(scores, basis, row_totals)


def _regularize_nndsvd_initialization(
    scores: FloatArray, basis: FloatArray, row_totals: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Apply the frozen relative floor to scores only; the Dirichlet update smooths B."""

    expected_scores = (SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT, SCIPLEX3_CANDIDATE_FACTOR_COUNT)
    expected_basis = (SCIPLEX3_CANDIDATE_FACTOR_COUNT, SCIPLEX3_FEATURE_COUNT)
    if (
        scores.shape != expected_scores
        or basis.shape != expected_basis
        or row_totals.shape != (SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,)
        or not bool(np.all(np.isfinite(scores)))
        or not bool(np.all(np.isfinite(basis)))
        or bool(np.any(scores < 0.0))
        or bool(np.any(basis < 0.0))
        or bool(np.any(row_totals <= 0.0))
    ):
        raise SciPlex3CandidateError("NNDSVD positivity-floor inputs are invalid")
    floor = (
        SCIPLEX3_CANDIDATE_NNDSVD_SCORE_FLOOR_FRACTION
        * row_totals[:, None]
        / SCIPLEX3_CANDIDATE_FACTOR_COUNT
    )
    positive_scores = np.maximum(np.asarray(scores, dtype=np.float64), floor)
    positive_scores *= row_totals[:, None] / np.sum(positive_scores, axis=1, keepdims=True)
    unchanged_basis = np.asarray(basis, dtype=np.float64)
    if bool(np.any(positive_scores <= 0.0)) or not bool(
        np.allclose(
            np.sum(positive_scores, axis=1),
            row_totals,
            rtol=8.0 * np.finfo(np.float64).eps,
            atol=0.0,
        )
    ):
        raise SciPlex3CandidateError("NNDSVD score smoothing failed")
    return positive_scores, unchanged_basis


@dataclass(slots=True)
class _LocalVariationalState:
    theta_shape: FloatArray
    theta_rate: FloatArray


@dataclass(frozen=True, slots=True)
class _PassSufficientStatistics:
    loading_counts: FloatArray
    well_theta_means: FloatArray
    allocation_entropy: float
    poisson_factorial: float
    theta_count_elog: float
    maximum_inner_sweeps: int
    maximum_terminal_shape_residual: float
    maximum_terminal_elog_residual: float
    inner_sweep_count_histogram: tuple[int, ...]


def _well_factor_means(
    alpha: FloatArray,
    rho: FloatArray,
    delta: FloatArray,
    validated: _ValidatedTrainingDesign,
) -> FloatArray:
    return _reconstruct_mean_activation(
        alpha,
        rho,
        delta,
        validated.training_well_plate_indices,
        validated.action_well_indices,
        validated.vehicle_well_indices,
    )


def _v5_action_parameters(
    alpha: FloatArray, rho: FloatArray, delta: FloatArray
) -> SciPlex3V5ActionParameters:
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_rho = np.log(rho)
        return SciPlex3V5ActionParameters(alpha=alpha, log_rho=log_rho, delta=delta)
    except SciPlex3V5ObjectiveError as error:
        raise SciPlex3CandidateError("candidate v5 action parameters are invalid") from error


def _update_action_block(
    well_theta_means: FloatArray,
    validated: _ValidatedTrainingDesign,
    *,
    initial: tuple[FloatArray, FloatArray, FloatArray] | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    if well_theta_means.shape != (
        SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,
        SCIPLEX3_CANDIDATE_FACTOR_COUNT,
    ) or not bool(np.all(np.isfinite(well_theta_means))):
        raise SciPlex3CandidateError("posterior well factor means are invalid")
    try:
        objective_design = SciPlex3V5Design(
            training_well_plate_indices=validated.training_well_plate_indices,
            action_well_indices=validated.action_well_indices,
            vehicle_well_indices=validated.vehicle_well_indices,
        )
        initial_parameters = None
        if initial is not None:
            initial_alpha, initial_rho, initial_delta = initial
            initial_parameters = _v5_action_parameters(initial_alpha, initial_rho, initial_delta)
        fitted = fit_fixed_q_action_context_m_step(
            well_theta_means,
            objective_design,
            initial=initial_parameters,
        )
    except SciPlex3V5ObjectiveError as error:
        raise SciPlex3CandidateError("candidate v5 action/context M-step failed") from error
    alpha = np.asarray(fitted.parameters.alpha, dtype=np.float64).copy()
    rho = np.exp(fitted.parameters.log_rho)
    delta = np.asarray(fitted.parameters.delta, dtype=np.float64).copy()
    means = _well_factor_means(alpha, rho, delta, validated)
    return alpha, rho, delta, means


def _initialize_local_state(
    validated: _ValidatedTrainingDesign, initial_well_scores: FloatArray
) -> _LocalVariationalState:
    record_count = validated.record_count
    theta_shape = np.empty((record_count, SCIPLEX3_CANDIDATE_FACTOR_COUNT), dtype=np.float64)
    theta_rate = np.empty_like(theta_shape)
    offset = 0
    tiny = np.finfo(np.float64).tiny
    for well_index, well in enumerate(validated.wells):
        stop = offset + well.counts.row_count
        means = np.maximum(initial_well_scores[well_index], tiny)
        theta_shape[offset:stop] = SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE
        theta_rate[offset:stop] = SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE / means + 1.0
        offset = stop
    return _LocalVariationalState(theta_shape, theta_rate)


def _cavi_pass(
    validated: _ValidatedTrainingDesign,
    state: _LocalVariationalState,
    loading_concentration: FloatArray,
    well_factor_means: FloatArray,
) -> _PassSufficientStatistics:
    loading_sums = np.sum(loading_concentration, axis=1)
    expected_log_loading = digamma(loading_concentration) - digamma(loading_sums)[:, None]
    if not bool(np.all(np.isfinite(expected_log_loading))):
        raise SciPlex3CandidateError("CAVI expected log loading is nonfinite")
    loading_counts = np.zeros_like(loading_concentration)
    well_theta_means = np.empty_like(well_factor_means)
    allocation_entropy = 0.0
    poisson_factorial = 0.0
    theta_count_elog = 0.0
    maximum_inner_sweeps = 0
    maximum_terminal_shape_residual = 0.0
    maximum_terminal_elog_residual = 0.0
    inner_sweep_count_histogram = [0] * SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
    expected_weighted_count_mass_terms: list[float] = []
    mass_tolerance_multiplier = SCIPLEX3_CANDIDATE_MASS_EPS_MULTIPLIER * np.finfo(np.float64).eps
    offset = 0
    for well_index, well in enumerate(validated.wells):
        row_count = well.counts.row_count
        omega = SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT / (
            SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT * row_count
        )
        well_theta_sum = np.zeros(SCIPLEX3_CANDIDATE_FACTOR_COUNT, dtype=np.float64)
        for batch_start in range(0, row_count, SCIPLEX3_CANDIDATE_BATCH_SIZE):
            batch_stop = min(batch_start + SCIPLEX3_CANDIDATE_BATCH_SIZE, row_count)
            global_slice = slice(offset + batch_start, offset + batch_stop)
            local_start = int(well.counts.indptr[batch_start])
            local_stop = int(well.counts.indptr[batch_stop])
            batch_indptr = well.counts.indptr[batch_start : batch_stop + 1] - local_start
            row_for_entry = np.repeat(
                np.arange(batch_stop - batch_start, dtype=np.int64), np.diff(batch_indptr)
            )
            feature_indices = well.counts.feature_indices[local_start:local_stop]
            values = well.counts.values[local_start:local_stop].astype(np.float64)
            prior_mean = well_factor_means[well_index]
            fixed_rate = SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE / prior_mean + 1.0
            batch_rate = np.broadcast_to(
                fixed_rate,
                (batch_stop - batch_start, SCIPLEX3_CANDIDATE_FACTOR_COUNT),
            )
            if not bool(np.all(np.isfinite(batch_rate))) or bool(np.any(batch_rate <= 1.0)):
                raise SciPlex3CandidateError("CAVI fixed factor rate is invalid")
            state.theta_rate[global_slice] = batch_rate
            passing_streak = 0
            final_allocated_counts: FloatArray | None = None
            final_responsibilities: FloatArray | None = None
            final_allocations: FloatArray | None = None
            terminal_shape_residual = math.inf
            terminal_elog_residual = math.inf
            sweep_count = 0
            for sweep_count in range(1, SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS + 1):
                current_shape = np.asarray(
                    state.theta_shape[global_slice], dtype=np.float64, order="C"
                )
                if not bool(np.all(np.isfinite(current_shape))) or bool(
                    np.any(current_shape <= 0.0)
                ):
                    raise SciPlex3CandidateError("CAVI warm-start factor shape is invalid")
                expected_log_theta = digamma(current_shape) - np.log(batch_rate)
                allocated_counts = np.zeros(
                    (batch_stop - batch_start, SCIPLEX3_CANDIDATE_FACTOR_COUNT),
                    dtype=np.float64,
                )
                responsibilities = np.empty(
                    (len(values), SCIPLEX3_CANDIDATE_FACTOR_COUNT), dtype=np.float64
                )
                allocations = np.empty_like(responsibilities)
                if len(values):
                    logits = (
                        expected_log_theta[row_for_entry]
                        + expected_log_loading[:, feature_indices].T
                    )
                    if not bool(np.all(np.isfinite(logits))):
                        raise SciPlex3CandidateError("CAVI allocation logits are nonfinite")
                    logits -= np.max(logits, axis=1, keepdims=True)
                    responsibilities[:] = np.exp(logits)
                    responsibility_sums = np.sum(responsibilities, axis=1, keepdims=True)
                    responsibilities /= responsibility_sums
                    normalized_sums = np.sum(responsibilities, axis=1)
                    if (
                        not bool(np.all(np.isfinite(responsibilities)))
                        or bool(np.any(responsibilities < 0.0))
                        or bool(np.any(np.abs(normalized_sums - 1.0) > mass_tolerance_multiplier))
                    ):
                        raise SciPlex3CandidateError(
                            "CAVI responsibility mass exceeds the frozen tolerance"
                        )
                    allocations[:] = values[:, None] * responsibilities
                    np.add.at(allocated_counts, row_for_entry, allocations)
                    row_input_mass = np.bincount(
                        row_for_entry,
                        weights=values,
                        minlength=batch_stop - batch_start,
                    )
                    row_allocated_mass = np.sum(allocated_counts, axis=1)
                    row_mass_tolerance = mass_tolerance_multiplier * np.maximum(1.0, row_input_mass)
                    if (
                        not bool(np.all(np.isfinite(allocations)))
                        or bool(np.any(allocations < 0.0))
                        or bool(
                            np.any(np.abs(row_allocated_mass - row_input_mass) > row_mass_tolerance)
                        )
                    ):
                        raise SciPlex3CandidateError(
                            "CAVI allocated count mass exceeds the frozen tolerance"
                        )
                proposal_shape = SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE + allocated_counts
                if not bool(np.all(np.isfinite(proposal_shape))) or bool(
                    np.any(proposal_shape < SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE)
                ):
                    raise SciPlex3CandidateError("CAVI factor-shape proposal is invalid")
                denominator = np.maximum(
                    1.0,
                    np.maximum(np.abs(current_shape), np.abs(proposal_shape)),
                )
                terminal_shape_residual = float(
                    np.max(np.abs(proposal_shape - current_shape) / denominator)
                )
                terminal_elog_residual = float(
                    np.max(np.abs(digamma(proposal_shape) - digamma(current_shape)))
                )
                if not math.isfinite(terminal_shape_residual) or not math.isfinite(
                    terminal_elog_residual
                ):
                    raise SciPlex3CandidateError("CAVI inner fixed-point residual is nonfinite")
                state.theta_shape[global_slice] = proposal_shape
                if (
                    terminal_shape_residual <= SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL
                    and terminal_elog_residual <= SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL
                ):
                    passing_streak += 1
                else:
                    passing_streak = 0
                if (
                    sweep_count >= SCIPLEX3_CANDIDATE_MIN_INNER_SWEEPS
                    and passing_streak >= SCIPLEX3_CANDIDATE_INNER_CONVERGENCE_STREAK
                ):
                    final_allocated_counts = allocated_counts
                    final_responsibilities = responsibilities
                    final_allocations = allocations
                    break
            if (
                final_allocated_counts is None
                or final_responsibilities is None
                or final_allocations is None
            ):
                raise SciPlex3CandidateError(
                    "candidate inner CAVI failed to converge within 50 sweeps"
                )
            maximum_inner_sweeps = max(maximum_inner_sweeps, sweep_count)
            maximum_terminal_shape_residual = max(
                maximum_terminal_shape_residual, terminal_shape_residual
            )
            maximum_terminal_elog_residual = max(
                maximum_terminal_elog_residual, terminal_elog_residual
            )
            inner_sweep_count_histogram[sweep_count - 1] += 1
            if len(values):
                expected_weighted_count_mass_terms.append(
                    omega * math.fsum(float(value) for value in values)
                )
                positive = final_responsibilities > 0.0
                allocation_entropy -= omega * float(
                    np.sum(final_allocations[positive] * np.log(final_responsibilities[positive]))
                )
                poisson_factorial += omega * float(np.sum(gammaln(values + 1.0)))
                for factor_index in range(SCIPLEX3_CANDIDATE_FACTOR_COUNT):
                    loading_counts[factor_index] += omega * np.bincount(
                        feature_indices,
                        weights=final_allocations[:, factor_index],
                        minlength=SCIPLEX3_FEATURE_COUNT,
                    )
            final_theta_shape = state.theta_shape[global_slice]
            theta_mean = final_theta_shape / batch_rate
            theta_elog = digamma(final_theta_shape) - np.log(batch_rate)
            well_theta_sum += np.sum(theta_mean, axis=0)
            theta_count_elog += omega * float(np.sum(final_allocated_counts * theta_elog))
        well_theta_means[well_index] = well_theta_sum / row_count
        offset += row_count
    if offset != validated.record_count:
        raise SciPlex3CandidateError("CAVI did not exhaust the exact p1 record stream")
    expected_weighted_count_mass = math.fsum(expected_weighted_count_mass_terms)
    actual_weighted_count_mass = math.fsum(
        float(value) for value in loading_counts.ravel(order="C")
    )
    global_mass_tolerance = mass_tolerance_multiplier * max(1.0, expected_weighted_count_mass)
    if (
        not math.isfinite(expected_weighted_count_mass)
        or not math.isfinite(actual_weighted_count_mass)
        or abs(actual_weighted_count_mass - expected_weighted_count_mass) > global_mass_tolerance
    ):
        raise SciPlex3CandidateError(
            "CAVI global equal-well-weighted count mass exceeds the frozen tolerance"
        )
    if (
        not all(
            math.isfinite(value)
            for value in (
                allocation_entropy,
                poisson_factorial,
                theta_count_elog,
                maximum_terminal_shape_residual,
                maximum_terminal_elog_residual,
            )
        )
        or not bool(np.all(np.isfinite(loading_counts)))
        or not bool(np.all(np.isfinite(well_theta_means)))
    ):
        raise SciPlex3CandidateError("CAVI sufficient statistics are nonfinite")
    return _PassSufficientStatistics(
        loading_counts=loading_counts,
        well_theta_means=well_theta_means,
        allocation_entropy=allocation_entropy,
        poisson_factorial=poisson_factorial,
        theta_count_elog=theta_count_elog,
        maximum_inner_sweeps=maximum_inner_sweeps,
        maximum_terminal_shape_residual=maximum_terminal_shape_residual,
        maximum_terminal_elog_residual=maximum_terminal_elog_residual,
        inner_sweep_count_histogram=tuple(inner_sweep_count_histogram),
    )


def _gamma_entropy(shape: FloatArray, rate: FloatArray) -> FloatArray:
    return np.asarray(
        shape - np.log(rate) + gammaln(shape) + (1.0 - shape) * digamma(shape),
        dtype=np.float64,
    )


def _elbo(
    validated: _ValidatedTrainingDesign,
    state: _LocalVariationalState,
    sufficient: _PassSufficientStatistics,
    loading_concentration: FloatArray,
    action_parameters: SciPlex3V5ActionParameters,
) -> float:
    if type(action_parameters) is not SciPlex3V5ActionParameters:
        raise SciPlex3CandidateError("candidate tracked ELBO requires exact v5 action parameters")
    factor_prior_shape = SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE
    loading_sum = np.sum(loading_concentration, axis=1)
    expected_log_loading = digamma(loading_concentration) - digamma(loading_sum)[:, None]
    likelihood = (
        sufficient.theta_count_elog
        + float(np.sum(sufficient.loading_counts * expected_log_loading))
        + sufficient.allocation_entropy
        - sufficient.poisson_factorial
    )
    local_terms = 0.0
    posterior_well_factor_means = np.empty(
        (SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT, SCIPLEX3_CANDIDATE_FACTOR_COUNT),
        dtype=np.float64,
    )
    offset = 0
    for well_index, well in enumerate(validated.wells):
        stop = offset + well.counts.row_count
        omega = SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT / (
            SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT * well.counts.row_count
        )
        theta_shape = state.theta_shape[offset:stop]
        theta_rate = state.theta_rate[offset:stop]
        theta_mean = theta_shape / theta_rate
        theta_elog = digamma(theta_shape) - np.log(theta_rate)
        likelihood -= omega * float(np.sum(theta_mean))
        posterior_well_factor_means[well_index] = np.mean(theta_mean, axis=0)
        theta_prior = (
            factor_prior_shape * math.log(factor_prior_shape)
            - float(gammaln(factor_prior_shape))
            + (factor_prior_shape - 1.0) * theta_elog
        )
        local_terms += omega * float(
            np.sum(theta_prior) + np.sum(_gamma_entropy(theta_shape, theta_rate))
        )
        offset = stop
    if offset != validated.record_count:
        raise SciPlex3CandidateError("candidate tracked ELBO did not exhaust the record stream")
    feature_count = float(SCIPLEX3_FEATURE_COUNT)
    prior = SCIPLEX3_CANDIDATE_DIRICHLET_CONCENTRATION
    loading_prior = SCIPLEX3_CANDIDATE_FACTOR_COUNT * (
        float(gammaln(feature_count * prior)) - feature_count * float(gammaln(prior))
    ) + (prior - 1.0) * float(np.sum(expected_log_loading))
    log_beta = np.sum(gammaln(loading_concentration), axis=1) - gammaln(loading_sum)
    loading_entropy = float(
        np.sum(
            log_beta
            + (loading_sum - feature_count) * digamma(loading_sum)
            - np.sum((loading_concentration - 1.0) * digamma(loading_concentration), axis=1)
        )
    )
    try:
        action_context = fixed_q_full_elbo_action_context(
            posterior_well_factor_means,
            action_parameters,
            SciPlex3V5Design(
                training_well_plate_indices=validated.training_well_plate_indices,
                action_well_indices=validated.action_well_indices,
                vehicle_well_indices=validated.vehicle_well_indices,
            ),
        )
    except SciPlex3V5ObjectiveError as error:
        raise SciPlex3CandidateError("candidate tracked v5 action objective failed") from error
    result = likelihood + local_terms + loading_prior + loading_entropy + action_context
    if not math.isfinite(result):
        raise SciPlex3CandidateError("candidate tracked ELBO is nonfinite")
    return float(result)


def _factor_contributions(mean_activation: FloatArray) -> FloatArray:
    if mean_activation.shape != (
        SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,
        SCIPLEX3_CANDIDATE_FACTOR_COUNT,
    ):
        raise SciPlex3CandidateError("candidate mean activation shape is invalid")
    canonical = _canonical_audit_matrix(mean_activation)
    contributions = np.asarray(
        [
            math.fsum(
                float(canonical[row_index, factor_index])
                for row_index in range(SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT)
            )
            / SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT
            for factor_index in range(SCIPLEX3_CANDIDATE_FACTOR_COUNT)
        ],
        dtype=np.float64,
    )
    if not bool(np.all(np.isfinite(contributions))) or bool(np.any(contributions <= 0.0)):
        raise SciPlex3CandidateError("candidate factor contribution is invalid")
    return contributions


def _canonical_factor_order(basis: FloatArray, contributions: FloatArray) -> IntArray:
    if basis.shape != (SCIPLEX3_CANDIDATE_FACTOR_COUNT, SCIPLEX3_FEATURE_COUNT):
        raise SciPlex3CandidateError("candidate factor-order basis shape is invalid")
    rounded_contributions = np.round(contributions, SCIPLEX3_CANDIDATE_FACTOR_ORDER_DECIMALS)
    if contributions.shape != (SCIPLEX3_CANDIDATE_FACTOR_COUNT,) or not bool(
        np.all(np.isfinite(rounded_contributions))
    ):
        raise SciPlex3CandidateError("candidate contains invalid factor contributions")
    canonical_basis = _canonical_audit_matrix(basis)
    digests = tuple(
        _sha256_bytes(np.asarray(canonical_basis[index], dtype="<f8", order="C").tobytes(order="C"))
        for index in range(SCIPLEX3_CANDIDATE_FACTOR_COUNT)
    )
    keys = tuple(
        (-float(rounded_contributions[index]), digests[index])
        for index in range(SCIPLEX3_CANDIDATE_FACTOR_COUNT)
    )
    if len(set(keys)) != SCIPLEX3_CANDIDATE_FACTOR_COUNT:
        raise SciPlex3CandidateError("candidate contains duplicate canonical factor keys")
    return np.asarray(
        sorted(range(SCIPLEX3_CANDIDATE_FACTOR_COUNT), key=keys.__getitem__), dtype=np.int64
    )


def _fit_sciplex3_candidate_exact(
    training: P1TrainingData, design: SciPlex3P1DesignBindings
) -> SciPlex3GammaPoissonCandidate:
    validated = _validate_training_design(training, design)
    well_means = np.stack([well.counts.feature_mean() for well in validated.wells])
    initial_scores, initial_basis = _nndsvd_initialization(well_means)
    alpha, rho, delta, well_factor_means = _update_action_block(initial_scores, validated)
    loading_concentration = (
        SCIPLEX3_CANDIDATE_DIRICHLET_CONCENTRATION
        + SCIPLEX3_CANDIDATE_LAMBDA_INITIAL_MASS * initial_basis
    )
    state = _initialize_local_state(validated, initial_scores)
    previous_sufficient = _cavi_pass(
        validated,
        state,
        loading_concentration,
        well_factor_means,
    )
    previous_elbo = _elbo(
        validated,
        state,
        previous_sufficient,
        loading_concentration,
        _v5_action_parameters(alpha, rho, delta),
    )
    initial_basis_mean = loading_concentration / np.sum(
        loading_concentration, axis=1, keepdims=True
    )
    initial_order = _canonical_factor_order(
        initial_basis_mean, _factor_contributions(well_factor_means)
    )
    initial_equilibration = SciPlex3CandidateInitialEquilibration(
        elbo=float(previous_elbo),
        factor_order=tuple(int(value) for value in initial_order),
        maximum_inner_sweeps=previous_sufficient.maximum_inner_sweeps,
        maximum_terminal_shape_residual=(previous_sufficient.maximum_terminal_shape_residual),
        maximum_terminal_elog_residual=(previous_sufficient.maximum_terminal_elog_residual),
        inner_sweep_count_histogram=previous_sufficient.inner_sweep_count_histogram,
    )
    trace: list[SciPlex3CandidateTraceEntry] = []
    for iteration in range(1, SCIPLEX3_CANDIDATE_MAX_OUTER_ITERATIONS + 1):
        loading_concentration = (
            SCIPLEX3_CANDIDATE_DIRICHLET_CONCENTRATION + previous_sufficient.loading_counts
        )
        alpha, rho, delta, well_factor_means = _update_action_block(
            previous_sufficient.well_theta_means,
            validated,
            initial=(alpha, rho, delta),
        )
        current_sufficient = _cavi_pass(
            validated,
            state,
            loading_concentration,
            well_factor_means,
        )
        current_elbo = _elbo(
            validated,
            state,
            current_sufficient,
            loading_concentration,
            _v5_action_parameters(alpha, rho, delta),
        )
        current_basis = loading_concentration / np.sum(loading_concentration, axis=1, keepdims=True)
        current_contributions = _factor_contributions(well_factor_means)
        current_order = _canonical_factor_order(current_basis, current_contributions)
        change = current_elbo - previous_elbo
        decrease_tolerance = SCIPLEX3_CANDIDATE_ELBO_DECREASE_RTOL * max(1.0, abs(previous_elbo))
        if change < -decrease_tolerance:
            raise SciPlex3CandidateError("candidate tracked ELBO materially decreased")
        relative_change = abs(change) / max(1.0, abs(previous_elbo))
        trace.append(
            SciPlex3CandidateTraceEntry(
                iteration=iteration,
                elbo=float(current_elbo),
                relative_change=float(relative_change),
                factor_order=tuple(int(value) for value in current_order),
                maximum_inner_sweeps=current_sufficient.maximum_inner_sweeps,
                maximum_terminal_shape_residual=(
                    current_sufficient.maximum_terminal_shape_residual
                ),
                maximum_terminal_elog_residual=(current_sufficient.maximum_terminal_elog_residual),
                inner_sweep_count_histogram=(current_sufficient.inner_sweep_count_histogram),
            )
        )
        previous_elbo = current_elbo
        previous_sufficient = current_sufficient
        if iteration >= SCIPLEX3_CANDIDATE_MIN_OUTER_ITERATIONS:
            terminal = trace[-SCIPLEX3_CANDIDATE_CONVERGENCE_STREAK:]
            if (
                all(
                    item.relative_change <= SCIPLEX3_CANDIDATE_CONVERGENCE_RTOL for item in terminal
                )
                and len({item.factor_order for item in terminal}) == 1
            ):
                break
    else:
        raise SciPlex3CandidateError("candidate CAVI failed to converge within 50 outer passes")
    loading_sums = np.sum(loading_concentration, axis=1)
    basis = loading_concentration / loading_sums[:, None]
    contributions = _factor_contributions(well_factor_means)
    order = _canonical_factor_order(basis, contributions)
    basis = basis[order]
    alpha = alpha[order]
    rho = rho[:, order]
    delta = delta[:, :, order]
    mean_activation = _canonical_audit_matrix(
        _reconstruct_mean_activation(
            alpha,
            rho,
            delta,
            validated.training_well_plate_indices,
            validated.action_well_indices,
            validated.vehicle_well_indices,
        )
    )
    contributions = _factor_contributions(mean_activation)
    candidate = SciPlex3GammaPoissonCandidate(
        ordered_feature_keys=training.ordered_feature_keys,
        compounds=design.compounds,
        plate_ids=design.plate_ids,
        training_well_ids=tuple(well.well_id for well in validated.wells),
        _basis=basis,
        _alpha=alpha,
        _rho=rho,
        _delta=delta,
        _factor_shape=np.asarray([SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE], dtype=np.float64),
        _factor_contributions=contributions,
        _mean_activation=mean_activation,
        _training_well_plate_indices=validated.training_well_plate_indices,
        _action_well_indices=validated.action_well_indices,
        _vehicle_well_indices=validated.vehicle_well_indices,
        initial_equilibration=initial_equilibration,
        trace=tuple(trace),
        training_summary=SciPlex3CandidateTrainingSummary(
            record_count=validated.record_count,
            well_count=len(validated.wells),
            zero_panel_record_count=validated.zero_panel_record_count,
            design_sha256=design.fingerprint,
            training_data_sha256=training_data_fingerprint(training),
            provenance="real-p1",
        ),
    )
    behavior = candidate.behavior_manifest()
    if behavior["fit_converged"] is not True or behavior["all_parameters_finite"] is not True:
        raise SciPlex3CandidateError("candidate failed final convergence or finiteness gate")
    return candidate


def _golden_initial_equilibration() -> SciPlex3CandidateInitialEquilibration:
    histogram = [0] * SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
    histogram[SCIPLEX3_CANDIDATE_MIN_INNER_SWEEPS - 1] = SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT
    return SciPlex3CandidateInitialEquilibration(
        elbo=-110.0,
        factor_order=tuple(range(SCIPLEX3_CANDIDATE_FACTOR_COUNT)),
        maximum_inner_sweeps=SCIPLEX3_CANDIDATE_MIN_INNER_SWEEPS,
        maximum_terminal_shape_residual=3e-12,
        maximum_terminal_elog_residual=4e-12,
        inner_sweep_count_histogram=tuple(histogram),
    )


def _golden_trace() -> tuple[SciPlex3CandidateTraceEntry, ...]:
    elbos = (
        -100.0,
        -95.0,
        -93.5,
        -93.1,
        -93.01,
        -93.001,
        -93.000010,
        -93.000005,
        -93.000001,
        -93.0,
    )
    factor_order = tuple(range(SCIPLEX3_CANDIDATE_FACTOR_COUNT))
    histogram = [0] * SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
    histogram[SCIPLEX3_CANDIDATE_MIN_INNER_SWEEPS - 1] = SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT
    entries: list[SciPlex3CandidateTraceEntry] = []
    previous_elbo = _golden_initial_equilibration().elbo
    for index, elbo in enumerate(elbos):
        relative = abs(elbo - previous_elbo) / max(1.0, abs(previous_elbo))
        entries.append(
            SciPlex3CandidateTraceEntry(
                iteration=index + 1,
                elbo=float(elbo),
                relative_change=float(relative),
                factor_order=factor_order,
                maximum_inner_sweeps=SCIPLEX3_CANDIDATE_MIN_INNER_SWEEPS,
                maximum_terminal_shape_residual=1e-12,
                maximum_terminal_elog_residual=2e-12,
                inner_sweep_count_histogram=tuple(histogram),
            )
        )
        previous_elbo = elbo
    return tuple(entries)


def build_sciplex3_synthetic_golden_candidate() -> SciPlex3GammaPoissonCandidate:
    """Build a fast full-topology exact-class fixture; it is never biological evidence."""

    factor = np.arange(1, SCIPLEX3_CANDIDATE_FACTOR_COUNT + 1, dtype=np.float64)[:, None]
    feature = np.arange(1, SCIPLEX3_FEATURE_COUNT + 1, dtype=np.float64)[None, :]
    basis = 1.0 + np.mod(factor * (feature + 7.0), 97.0)
    basis /= np.sum(basis, axis=1, keepdims=True)
    compounds = tuple(
        f"golden-compound-{index:03d}" for index in range(SCIPLEX3_CANDIDATE_COMPOUND_COUNT)
    )
    plates = tuple(f"golden-plate-{index}" for index in range(SCIPLEX3_CANDIDATE_PLATE_COUNT))
    rho_grid = np.exp(
        0.01
        * (np.arange(SCIPLEX3_CANDIDATE_PLATE_COUNT, dtype=np.float64)[:, None] - 3.5)
        * (1.0 + np.arange(SCIPLEX3_CANDIDATE_FACTOR_COUNT, dtype=np.float64)[None, :] / 16.0)
    )
    rho = _canonical_audit_matrix(rho_grid / np.mean(rho_grid, axis=0, keepdims=True))
    compound_index = np.arange(SCIPLEX3_CANDIDATE_COMPOUND_COUNT, dtype=np.float64)[:, None, None]
    dose_index = np.arange(len(SCIPLEX3_CANDIDATE_DOSES_NM), dtype=np.float64)[None, :, None]
    factor_index = np.arange(SCIPLEX3_CANDIDATE_FACTOR_COUNT, dtype=np.float64)[None, None, :]
    delta = _canonical_audit_matrix(
        0.035 * np.sin((compound_index + 1.0) * (dose_index + 1.0) * (factor_index + 1.0))
    )
    action_plate_indices = np.fromfunction(
        lambda compound, dose: (compound + dose) % SCIPLEX3_CANDIDATE_PLATE_COUNT,
        (SCIPLEX3_CANDIDATE_COMPOUND_COUNT, len(SCIPLEX3_CANDIDATE_DOSES_NM)),
        dtype=int,
    ).astype(np.int64)
    action_well_indices = np.arange(SCIPLEX3_CANDIDATE_ACTION_COUNT, dtype=np.int64).reshape(
        SCIPLEX3_CANDIDATE_COMPOUND_COUNT, len(SCIPLEX3_CANDIDATE_DOSES_NM)
    )
    vehicle_well_indices = (
        SCIPLEX3_CANDIDATE_ACTION_COUNT
        + np.arange(SCIPLEX3_CANDIDATE_CONTROL_WELL_COUNT, dtype=np.int64)
    ).reshape(SCIPLEX3_CANDIDATE_PLATE_COUNT, 2)
    training_well_plate_indices = np.empty(SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT, dtype=np.int64)
    training_well_plate_indices[action_well_indices] = action_plate_indices
    for plate_index in range(SCIPLEX3_CANDIDATE_PLATE_COUNT):
        training_well_plate_indices[vehicle_well_indices[plate_index]] = plate_index
    training_well_ids = tuple(
        [
            f"golden-action-{compound_index:03d}-{dose_index}"
            for compound_index in range(SCIPLEX3_CANDIDATE_COMPOUND_COUNT)
            for dose_index in range(len(SCIPLEX3_CANDIDATE_DOSES_NM))
        ]
        + [
            f"golden-vehicle-{plate_index}-{control_index}"
            for plate_index in range(SCIPLEX3_CANDIDATE_PLATE_COUNT)
            for control_index in range(2)
        ]
    )
    alpha = np.log(np.linspace(2.5, 0.75, SCIPLEX3_CANDIDATE_FACTOR_COUNT))
    mean_activation = _canonical_audit_matrix(
        _reconstruct_mean_activation(
            alpha,
            rho,
            delta,
            training_well_plate_indices,
            action_well_indices,
            vehicle_well_indices,
        )
    )
    synthetic_design = {
        "compound_count": SCIPLEX3_CANDIDATE_COMPOUND_COUNT,
        "doses_nm": list(SCIPLEX3_CANDIDATE_DOSES_NM),
        "plate_count": SCIPLEX3_CANDIDATE_PLATE_COUNT,
        "provenance": "synthetic-golden",
    }
    return SciPlex3GammaPoissonCandidate(
        ordered_feature_keys=tuple(
            f"golden-feature-{index:04d}" for index in range(SCIPLEX3_FEATURE_COUNT)
        ),
        compounds=compounds,
        plate_ids=plates,
        training_well_ids=training_well_ids,
        _basis=basis,
        _alpha=alpha,
        _rho=rho,
        _delta=delta,
        _factor_shape=np.asarray([SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE], dtype=np.float64),
        _factor_contributions=_factor_contributions(mean_activation),
        _mean_activation=mean_activation,
        _training_well_plate_indices=training_well_plate_indices,
        _action_well_indices=action_well_indices,
        _vehicle_well_indices=vehicle_well_indices,
        initial_equilibration=_golden_initial_equilibration(),
        trace=_golden_trace(),
        training_summary=SciPlex3CandidateTrainingSummary(
            record_count=SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT,
            well_count=SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT,
            zero_panel_record_count=SCIPLEX3_CANDIDATE_ZERO_PANEL_RECORD_COUNT,
            design_sha256=_canonical_json_sha256(synthetic_design),
            training_data_sha256=_canonical_json_sha256(
                {"fixture": "full-topology-no-biological-data-v5"}
            ),
            provenance="synthetic-golden",
        ),
    )


SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256: Final = (
    "e5f81e28b8f4efbf5cffd64afa326d380b4c071450fd02339b4a46102a2e70a2"
)
SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256: Final = (
    "9c364005b01142bcfe74a8400ab4b9209ed635b74a703167969aa5e8d80b9e2c"
)


def candidate_golden_model_bytes() -> bytes:
    """Build and verify the reference-runtime golden lazily, never during package import."""

    candidate = build_sciplex3_synthetic_golden_candidate()
    payload = candidate.canonical_model_bytes()
    if _sha256_bytes(payload) != SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256:
        raise SciPlex3CandidateError(
            "fresh synthetic golden model differs from the dual-runtime literal"
        )
    if _sha256_bytes(candidate.golden_sample().samples.tobytes(order="C")) != (
        SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256
    ):
        raise SciPlex3CandidateError(
            "fresh synthetic golden sample differs from the dual-runtime literal"
        )
    return payload


def verify_sciplex3_candidate_golden(candidate: SciPlex3GammaPoissonCandidate) -> bool:
    """Verify the fixed synthetic candidate and its internal p1-only sampling behavior."""

    if type(candidate) is not SciPlex3GammaPoissonCandidate:
        return False
    if candidate.model_artifact_sha256 != SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256:
        return False
    sample_sha256 = _sha256_bytes(candidate.golden_sample().samples.tobytes(order="C"))
    return sample_sha256 == SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256


def load_sciplex3_candidate(
    payload: bytes, *, expected_sha256: str
) -> SciPlex3GammaPoissonCandidate:
    """Module-level exact loader for the trusted factory registry."""

    return SciPlex3GammaPoissonCandidate.load_exact(payload, expected_sha256=expected_sha256)


def fit_sciplex3_candidate(
    training: P1TrainingData, design: SciPlex3P1DesignBindings
) -> SciPlex3GammaPoissonCandidate:
    """Module-level exact p1-only training entrypoint for the trusted workflow."""

    return SciPlex3GammaPoissonCandidate.fit(training, design)


__all__ = [
    "SCIPLEX3_CANDIDATE_ACTION_COUNT",
    "SCIPLEX3_CANDIDATE_COMPOUND_COUNT",
    "SCIPLEX3_CANDIDATE_DOSES_NM",
    "SCIPLEX3_CANDIDATE_FACTOR_COUNT",
    "SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE",
    "SCIPLEX3_CANDIDATE_GOLDEN_MODEL_SHA256",
    "SCIPLEX3_CANDIDATE_GOLDEN_SAMPLE_SHA256",
    "SCIPLEX3_CANDIDATE_IMPLEMENTATION_VERSION",
    "SCIPLEX3_CANDIDATE_MODEL_ID",
    "SCIPLEX3_CANDIDATE_MODEL_SCHEMA",
    "SCIPLEX3_CANDIDATE_MODEL_SCHEMA_VERSION",
    "SCIPLEX3_CANDIDATE_OUTPUT_MODEL_SCHEMA_SHA256",
    "SCIPLEX3_CANDIDATE_OUTPUT_MODEL_TOPOLOGY",
    "SCIPLEX3_CANDIDATE_REFERENCE_RUNTIME",
    "SCIPLEX3_CANDIDATE_SPECIFICATION_SCHEMA",
    "SCIPLEX3_CANDIDATE_SPECIFICATION_SCHEMA_VERSION",
    "SCIPLEX3_CANDIDATE_SPECIFICATION_SHA256",
    "SCIPLEX3_CANDIDATE_TAU_GRID",
    "SCIPLEX3_CANDIDATE_V5_NEUTRAL_CONTEXT_ID",
    "CandidateRawCountSamples",
    "CandidateSampleRequest",
    "SciPlex3CandidateError",
    "SciPlex3CandidateInitialEquilibration",
    "SciPlex3CandidateTraceEntry",
    "SciPlex3CandidateTrainingSummary",
    "SciPlex3GammaPoissonCandidate",
    "SciPlex3P1ActionBinding",
    "SciPlex3P1DesignBindings",
    "SciPlex3P1VehicleBinding",
    "build_sciplex3_synthetic_golden_candidate",
    "candidate_golden_model_bytes",
    "candidate_model_schema_manifest",
    "candidate_specification_manifest",
    "fit_sciplex3_candidate",
    "load_sciplex3_candidate",
    "training_data_fingerprint",
    "verify_sciplex3_candidate_golden",
]
