"""Source-free sci-Plex3 candidate-v5 action/context objective and M-step.

This module deliberately does not load sci-Plex3 source data and does not replace the historical
v4 candidate.  It freezes the mathematical core required before a v5 real-source execution can be
proposed: one equal-well fixed-q objective, its analytic derivatives, and a monotone action/context
M-step.

For fixed local variational factors, let ``t[w, k]`` be the posterior mean of factor ``k`` in well
``w`` and let ``m[w, k] = exp(eta[w, k])`` be its prior mean.  All action/context-dependent terms of
the full ELBO are, up to a fixed-q additive constant,

``Q = -(N/W) r sum_wk (eta[w,k] + t[w,k] exp(-eta[w,k])) - P(delta)``.

Here ``N=94785``, ``W=768``, ``r=0.1``, and ``P`` is the declared magnitude plus dose-curvature
penalty.  Consequently, differences in this objective are exactly the corresponding fixed-q full
ELBO differences.  V4's dose block omitted ``N/W``; every v5 objective and derivative below uses
the same exact scale.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SCIPLEX3_V5_OBJECTIVE_VERSION: Final = "5.0.0"
SCIPLEX3_V5_TRAINING_RECORD_COUNT: Final = 94_785
SCIPLEX3_V5_TRAINING_WELL_COUNT: Final = 768
SCIPLEX3_V5_PLATE_COUNT: Final = 8
SCIPLEX3_V5_FACTOR_COUNT: Final = 16
SCIPLEX3_V5_COMPOUND_COUNT: Final = 188
SCIPLEX3_V5_DOSE_COUNT: Final = 4
SCIPLEX3_V5_CONTROL_WELL_COUNT: Final = 16
SCIPLEX3_V5_FIXED_FACTOR_SHAPE: Final = 0.1
SCIPLEX3_V5_EQUAL_WELL_SCALE: Final = (
    SCIPLEX3_V5_TRAINING_RECORD_COUNT / SCIPLEX3_V5_TRAINING_WELL_COUNT
)
SCIPLEX3_V5_ACTION_LIKELIHOOD_SCALE: Final = (
    SCIPLEX3_V5_EQUAL_WELL_SCALE * SCIPLEX3_V5_FIXED_FACTOR_SHAPE
)
# A transformed alpha gradient can sum all eight plate-coordinate residuals.  The internal solve is
# nine times tighter, while this public gate remains above the representable objective plateau seen
# when a 768-well Newton improvement is below one float64 ulp.
SCIPLEX3_V5_GRADIENT_TOL: Final = 3e-4
SCIPLEX3_V5_DOSE_NEWTON_MAX_STEPS: Final = 50
SCIPLEX3_V5_BACKTRACK_MAX_STEPS: Final = 24
SCIPLEX3_V5_NUMERICAL_EPS_MULTIPLIER: Final = 128.0
_INTERNAL_GRADIENT_TOL: Final = SCIPLEX3_V5_GRADIENT_TOL / (SCIPLEX3_V5_PLATE_COUNT + 1.0)

_SECOND_DIFFERENCE: Final[FloatArray] = np.asarray(
    [[1.0, -2.0, 1.0, 0.0], [0.0, 1.0, -2.0, 1.0]], dtype=np.float64
)
_DOSE_PENALTY_HESSIAN: Final[FloatArray] = (
    np.eye(SCIPLEX3_V5_DOSE_COUNT, dtype=np.float64) / 4.0
    + _SECOND_DIFFERENCE.T @ _SECOND_DIFFERENCE
)


class SciPlex3V5ObjectiveError(ValueError):
    """Raised when the source-free v5 objective contract cannot be satisfied."""


def _freeze_float(value: object, *, shape: tuple[int, ...], name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind != "f":
        raise SciPlex3V5ObjectiveError(f"{name} must be a float array with shape {shape}")
    canonical = np.asarray(raw, dtype="<f8", order="C")
    if not bool(np.all(np.isfinite(canonical))):
        raise SciPlex3V5ObjectiveError(f"{name} must be finite")
    return np.frombuffer(canonical.tobytes(order="C"), dtype="<f8").reshape(shape)


def _freeze_int(value: object, *, shape: tuple[int, ...], name: str) -> IntArray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in {"i", "u"}:
        raise SciPlex3V5ObjectiveError(f"{name} must be an integer array with shape {shape}")
    if raw.dtype.kind == "u" and bool(np.any(raw > np.iinfo(np.int64).max)):
        raise SciPlex3V5ObjectiveError(f"{name} exceeds signed 64-bit support")
    canonical = np.asarray(raw, dtype="<i8", order="C")
    return np.frombuffer(canonical.tobytes(order="C"), dtype="<i8").reshape(shape)


def _objective_tolerance(value: float) -> float:
    return float(
        SCIPLEX3_V5_NUMERICAL_EPS_MULTIPLIER * np.finfo(np.float64).eps * max(1.0, abs(value))
    )


@dataclass(frozen=True, slots=True, eq=False)
class SciPlex3V5Design:
    """Exact outcome-free p1 well topology needed by the source-free objective."""

    training_well_plate_indices: IntArray
    action_well_indices: IntArray
    vehicle_well_indices: IntArray
    _compound_by_well: IntArray = field(init=False, repr=False)
    _dose_by_well: IntArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        plates = _freeze_int(
            self.training_well_plate_indices,
            shape=(SCIPLEX3_V5_TRAINING_WELL_COUNT,),
            name="v5 training-well plate indices",
        )
        actions = _freeze_int(
            self.action_well_indices,
            shape=(SCIPLEX3_V5_COMPOUND_COUNT, SCIPLEX3_V5_DOSE_COUNT),
            name="v5 action-well indices",
        )
        vehicles = _freeze_int(
            self.vehicle_well_indices,
            shape=(SCIPLEX3_V5_PLATE_COUNT, 2),
            name="v5 vehicle-well indices",
        )
        topology = np.concatenate((actions.ravel(order="C"), vehicles.ravel(order="C")))
        if (
            bool(np.any(plates < 0))
            or bool(np.any(plates >= SCIPLEX3_V5_PLATE_COUNT))
            or bool(np.any(topology < 0))
            or bool(np.any(topology >= SCIPLEX3_V5_TRAINING_WELL_COUNT))
            or not np.array_equal(
                np.sort(topology),
                np.arange(SCIPLEX3_V5_TRAINING_WELL_COUNT, dtype=np.int64),
            )
        ):
            raise SciPlex3V5ObjectiveError(
                "v5 action and vehicle topology must partition all 768 wells"
            )
        for plate_index in range(SCIPLEX3_V5_PLATE_COUNT):
            if not bool(np.all(plates[vehicles[plate_index]] == plate_index)):
                raise SciPlex3V5ObjectiveError(
                    "v5 vehicle wells must belong to their declared plate"
                )
        compound_by_well = np.full(SCIPLEX3_V5_TRAINING_WELL_COUNT, -1, dtype=np.int64)
        dose_by_well = np.full(SCIPLEX3_V5_TRAINING_WELL_COUNT, -1, dtype=np.int64)
        for compound_index in range(SCIPLEX3_V5_COMPOUND_COUNT):
            for dose_index in range(SCIPLEX3_V5_DOSE_COUNT):
                well_index = int(actions[compound_index, dose_index])
                compound_by_well[well_index] = compound_index
                dose_by_well[well_index] = dose_index
        object.__setattr__(self, "training_well_plate_indices", plates)
        object.__setattr__(self, "action_well_indices", actions)
        object.__setattr__(self, "vehicle_well_indices", vehicles)
        object.__setattr__(
            self,
            "_compound_by_well",
            _freeze_int(
                compound_by_well,
                shape=(SCIPLEX3_V5_TRAINING_WELL_COUNT,),
                name="v5 compound-by-well lookup",
            ),
        )
        object.__setattr__(
            self,
            "_dose_by_well",
            _freeze_int(
                dose_by_well,
                shape=(SCIPLEX3_V5_TRAINING_WELL_COUNT,),
                name="v5 dose-by-well lookup",
            ),
        )


def log_rho_from_feasible_coordinates(coordinates: object) -> FloatArray:
    """Map seven anchored coordinates per factor to the arithmetic-mean-one rho gauge."""

    free = _freeze_float(
        coordinates,
        shape=(SCIPLEX3_V5_PLATE_COUNT - 1, SCIPLEX3_V5_FACTOR_COUNT),
        name="v5 feasible log-rho coordinates",
    )
    anchored = np.vstack((free, np.zeros((1, SCIPLEX3_V5_FACTOR_COUNT), dtype=np.float64)))
    normalization = logsumexp(anchored, axis=0) - math.log(SCIPLEX3_V5_PLATE_COUNT)
    result = np.asarray(anchored - normalization[None, :], dtype=np.float64)
    if not bool(
        np.allclose(
            np.mean(np.exp(result), axis=0),
            1.0,
            rtol=0.0,
            atol=5e-13,
        )
    ):
        raise SciPlex3V5ObjectiveError("v5 feasible log-rho map lost its mean-one gauge")
    return result


def feasible_coordinates_from_log_rho(log_rho: object) -> FloatArray:
    """Return the unique seven-coordinate representation of a feasible log-rho tensor."""

    values = _validate_log_rho(log_rho)
    return np.asarray(values[:-1] - values[-1][None, :], dtype=np.float64)


def _validate_log_rho(value: object) -> FloatArray:
    log_rho = _freeze_float(
        value,
        shape=(SCIPLEX3_V5_PLATE_COUNT, SCIPLEX3_V5_FACTOR_COUNT),
        name="v5 log-rho",
    )
    log_mean = logsumexp(log_rho, axis=0) - math.log(SCIPLEX3_V5_PLATE_COUNT)
    rho = np.exp(log_rho)
    if (
        not bool(np.allclose(log_mean, 0.0, rtol=0.0, atol=5e-13))
        or not bool(np.all(np.isfinite(rho)))
        or bool(np.any(rho <= 0.0))
        or bool(np.any(rho >= SCIPLEX3_V5_PLATE_COUNT))
    ):
        raise SciPlex3V5ObjectiveError(
            "v5 log-rho must be interior and have factorwise arithmetic-mean-one rho"
        )
    return log_rho


@dataclass(frozen=True, slots=True, eq=False)
class SciPlex3V5ActionParameters:
    """V5 action parameters in the externally specified alpha/log-rho/delta coordinates."""

    alpha: FloatArray
    log_rho: FloatArray
    delta: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "alpha",
            _freeze_float(
                self.alpha,
                shape=(SCIPLEX3_V5_FACTOR_COUNT,),
                name="v5 alpha",
            ),
        )
        object.__setattr__(self, "log_rho", _validate_log_rho(self.log_rho))
        object.__setattr__(
            self,
            "delta",
            _freeze_float(
                self.delta,
                shape=(
                    SCIPLEX3_V5_COMPOUND_COUNT,
                    SCIPLEX3_V5_DOSE_COUNT,
                    SCIPLEX3_V5_FACTOR_COUNT,
                ),
                name="v5 dose effects",
            ),
        )


def action_context_parameter_sha256(parameters: SciPlex3V5ActionParameters) -> str:
    """Bind one complete immutable v5 action/context state to canonical float64 bytes."""

    if type(parameters) is not SciPlex3V5ActionParameters:
        raise SciPlex3V5ObjectiveError("v5 parameter identity requires exact action parameters")
    digest = hashlib.sha256(b"sciplex3-v5-action-context-state\0")
    for name, value in (
        (b"alpha", parameters.alpha),
        (b"log-rho", parameters.log_rho),
        (b"delta", parameters.delta),
    ):
        digest.update(name)
        digest.update(b"\0")
        digest.update(np.asarray(value, dtype="<f8", order="C").tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_posterior_means(value: object) -> FloatArray:
    means = _freeze_float(
        value,
        shape=(SCIPLEX3_V5_TRAINING_WELL_COUNT, SCIPLEX3_V5_FACTOR_COUNT),
        name="v5 fixed-q posterior well-factor means",
    )
    if bool(np.any(means <= 0.0)):
        raise SciPlex3V5ObjectiveError(
            "v5 fixed-q posterior well-factor means must be strictly positive"
        )
    return means


def _plate_intercepts(parameters: SciPlex3V5ActionParameters) -> FloatArray:
    return np.asarray(parameters.alpha[None, :] + parameters.log_rho, dtype=np.float64)


def _parameters_from_plate_intercepts(
    plate_intercepts: FloatArray, delta: FloatArray
) -> SciPlex3V5ActionParameters:
    beta = _freeze_float(
        plate_intercepts,
        shape=(SCIPLEX3_V5_PLATE_COUNT, SCIPLEX3_V5_FACTOR_COUNT),
        name="v5 plate log-intercepts",
    )
    alpha = np.asarray(
        logsumexp(beta, axis=0) - math.log(SCIPLEX3_V5_PLATE_COUNT),
        dtype=np.float64,
    )
    log_rho = np.asarray(beta - alpha[None, :], dtype=np.float64)
    return SciPlex3V5ActionParameters(alpha=alpha, log_rho=log_rho, delta=delta)


def _linear_predictor(
    parameters: SciPlex3V5ActionParameters, design: SciPlex3V5Design
) -> FloatArray:
    beta = _plate_intercepts(parameters)
    eta = np.asarray(beta[design.training_well_plate_indices], dtype=np.float64).copy()
    eta[design.action_well_indices.ravel(order="C")] += parameters.delta.reshape(
        SCIPLEX3_V5_COMPOUND_COUNT * SCIPLEX3_V5_DOSE_COUNT,
        SCIPLEX3_V5_FACTOR_COUNT,
    )
    if not bool(np.all(np.isfinite(eta))):
        raise SciPlex3V5ObjectiveError("v5 action/context linear predictor is nonfinite")
    return eta


def _dose_penalty(delta: FloatArray) -> float:
    second = np.einsum("sd,cdk->csk", _SECOND_DIFFERENCE, delta)
    value = float(np.sum(np.square(delta))) / 8.0 + 0.5 * float(np.sum(np.square(second)))
    if not math.isfinite(value):
        raise SciPlex3V5ObjectiveError("v5 dose penalty is nonfinite")
    return value


def fixed_q_full_elbo_action_context(
    posterior_well_factor_means: object,
    parameters: SciPlex3V5ActionParameters,
    design: SciPlex3V5Design,
) -> float:
    """Evaluate the canonical action/context-dependent fixed-q full ELBO terms."""

    if type(parameters) is not SciPlex3V5ActionParameters:
        raise SciPlex3V5ObjectiveError("v5 objective requires exact action parameters")
    if type(design) is not SciPlex3V5Design:
        raise SciPlex3V5ObjectiveError("v5 objective requires the exact design type")
    posterior = _validate_posterior_means(posterior_well_factor_means)
    eta = _linear_predictor(parameters, design)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        gamma_terms = eta + posterior * np.exp(-eta)
    value = -SCIPLEX3_V5_ACTION_LIKELIHOOD_SCALE * float(np.sum(gamma_terms)) - _dose_penalty(
        parameters.delta
    )
    if not math.isfinite(value):
        raise SciPlex3V5ObjectiveError("v5 fixed-q full-ELBO action objective is nonfinite")
    return float(value)


def independent_fixed_q_full_elbo_action_context(
    posterior_well_factor_means: object,
    parameters: SciPlex3V5ActionParameters,
    design: SciPlex3V5Design,
) -> float:
    """Independently evaluate the same objective with scalar topology traversal and ``fsum``."""

    if type(parameters) is not SciPlex3V5ActionParameters:
        raise SciPlex3V5ObjectiveError("independent v5 objective requires exact parameters")
    if type(design) is not SciPlex3V5Design:
        raise SciPlex3V5ObjectiveError("independent v5 objective requires exact design")
    posterior = _validate_posterior_means(posterior_well_factor_means)
    terms: list[float] = []
    try:
        for well_index in range(SCIPLEX3_V5_TRAINING_WELL_COUNT):
            plate_index = int(design.training_well_plate_indices[well_index])
            compound_index = int(design._compound_by_well[well_index])
            dose_index = int(design._dose_by_well[well_index])
            for factor_index in range(SCIPLEX3_V5_FACTOR_COUNT):
                eta = float(parameters.alpha[factor_index]) + float(
                    parameters.log_rho[plate_index, factor_index]
                )
                if compound_index >= 0:
                    eta += float(parameters.delta[compound_index, dose_index, factor_index])
                terms.append(eta + float(posterior[well_index, factor_index]) * math.exp(-eta))
        penalty_terms = [
            float(parameters.delta[compound, dose, factor]) ** 2 / 8.0
            for compound in range(SCIPLEX3_V5_COMPOUND_COUNT)
            for dose in range(SCIPLEX3_V5_DOSE_COUNT)
            for factor in range(SCIPLEX3_V5_FACTOR_COUNT)
        ]
        for compound in range(SCIPLEX3_V5_COMPOUND_COUNT):
            for factor in range(SCIPLEX3_V5_FACTOR_COUNT):
                block = parameters.delta[compound, :, factor]
                penalty_terms.append(0.5 * float(block[0] - 2.0 * block[1] + block[2]) ** 2)
                penalty_terms.append(0.5 * float(block[1] - 2.0 * block[2] + block[3]) ** 2)
        result = -SCIPLEX3_V5_ACTION_LIKELIHOOD_SCALE * math.fsum(terms) - math.fsum(penalty_terms)
    except OverflowError as error:
        raise SciPlex3V5ObjectiveError("independent v5 objective overflowed") from error
    if not math.isfinite(result):
        raise SciPlex3V5ObjectiveError("independent v5 objective is nonfinite")
    return float(result)


@dataclass(frozen=True, slots=True, eq=False)
class SciPlex3V5ObjectiveGradients:
    """Analytic gradients in alpha, feasible anchored log-rho, and delta coordinates."""

    alpha: FloatArray
    feasible_log_rho: FloatArray
    delta: FloatArray
    plate_intercepts: FloatArray

    @property
    def maximum_absolute(self) -> float:
        return max(
            float(np.max(np.abs(self.alpha))),
            float(np.max(np.abs(self.feasible_log_rho))),
            float(np.max(np.abs(self.delta))),
        )


def fixed_q_full_elbo_action_context_gradients(
    posterior_well_factor_means: object,
    parameters: SciPlex3V5ActionParameters,
    design: SciPlex3V5Design,
) -> SciPlex3V5ObjectiveGradients:
    """Return analytic gradients of :func:`fixed_q_full_elbo_action_context`."""

    if type(parameters) is not SciPlex3V5ActionParameters or type(design) is not SciPlex3V5Design:
        raise SciPlex3V5ObjectiveError("v5 gradients require exact parameters and design")
    posterior = _validate_posterior_means(posterior_well_factor_means)
    eta = _linear_predictor(parameters, design)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        eta_gradient = SCIPLEX3_V5_ACTION_LIKELIHOOD_SCALE * (posterior * np.exp(-eta) - 1.0)
    if not bool(np.all(np.isfinite(eta_gradient))):
        raise SciPlex3V5ObjectiveError("v5 action/context gradient is nonfinite")
    alpha_gradient = np.sum(eta_gradient, axis=0)
    plate_gradient = np.zeros((SCIPLEX3_V5_PLATE_COUNT, SCIPLEX3_V5_FACTOR_COUNT), dtype=np.float64)
    np.add.at(plate_gradient, design.training_well_plate_indices, eta_gradient)
    shares = np.exp(parameters.log_rho) / SCIPLEX3_V5_PLATE_COUNT
    feasible_gradient = plate_gradient[:-1] - shares[:-1] * np.sum(
        plate_gradient, axis=0, keepdims=True
    )
    delta_gradient = eta_gradient[design.action_well_indices.ravel(order="C")].reshape(
        SCIPLEX3_V5_COMPOUND_COUNT,
        SCIPLEX3_V5_DOSE_COUNT,
        SCIPLEX3_V5_FACTOR_COUNT,
    )
    penalty_gradient = parameters.delta / 4.0 + np.einsum(
        "ds,csk->cdk",
        _SECOND_DIFFERENCE.T,
        np.einsum("sd,cdk->csk", _SECOND_DIFFERENCE, parameters.delta),
    )
    delta_gradient = delta_gradient - penalty_gradient
    return SciPlex3V5ObjectiveGradients(
        alpha=_freeze_float(
            alpha_gradient,
            shape=(SCIPLEX3_V5_FACTOR_COUNT,),
            name="v5 alpha gradient",
        ),
        feasible_log_rho=_freeze_float(
            feasible_gradient,
            shape=(SCIPLEX3_V5_PLATE_COUNT - 1, SCIPLEX3_V5_FACTOR_COUNT),
            name="v5 feasible log-rho gradient",
        ),
        delta=_freeze_float(
            delta_gradient,
            shape=(
                SCIPLEX3_V5_COMPOUND_COUNT,
                SCIPLEX3_V5_DOSE_COUNT,
                SCIPLEX3_V5_FACTOR_COUNT,
            ),
            name="v5 dose gradient",
        ),
        plate_intercepts=_freeze_float(
            plate_gradient,
            shape=(SCIPLEX3_V5_PLATE_COUNT, SCIPLEX3_V5_FACTOR_COUNT),
            name="v5 plate-intercept gradient",
        ),
    )


def fixed_q_dose_block_gradient_hessian(
    posterior_means: object,
    plate_intercepts: object,
    delta: object,
) -> tuple[FloatArray, FloatArray]:
    """Return the canonical four-dose ELBO gradient and Hessian for one compound/factor."""

    posterior = _freeze_float(
        posterior_means,
        shape=(SCIPLEX3_V5_DOSE_COUNT,),
        name="v5 dose-block posterior means",
    )
    baseline = _freeze_float(
        plate_intercepts,
        shape=(SCIPLEX3_V5_DOSE_COUNT,),
        name="v5 dose-block plate intercepts",
    )
    effects = _freeze_float(
        delta,
        shape=(SCIPLEX3_V5_DOSE_COUNT,),
        name="v5 dose-block effects",
    )
    if bool(np.any(posterior <= 0.0)):
        raise SciPlex3V5ObjectiveError("v5 dose-block posterior means must be positive")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled = SCIPLEX3_V5_ACTION_LIKELIHOOD_SCALE * posterior * np.exp(-(baseline + effects))
    gradient = (
        scaled
        - SCIPLEX3_V5_ACTION_LIKELIHOOD_SCALE
        - (effects / 4.0 + _SECOND_DIFFERENCE.T @ (_SECOND_DIFFERENCE @ effects))
    )
    hessian = -np.diag(scaled) - _DOSE_PENALTY_HESSIAN
    if not bool(np.all(np.isfinite(gradient))) or not bool(np.all(np.isfinite(hessian))):
        raise SciPlex3V5ObjectiveError("v5 dose-block derivatives are nonfinite")
    return np.asarray(gradient, dtype=np.float64), np.asarray(hessian, dtype=np.float64)


def fixed_q_factor_arrowhead_gradient_hessian(
    posterior_well_means: object,
    plate_intercepts: object,
    delta: object,
    design: SciPlex3V5Design,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return one factor's canonical-Q gradient and full joint arrowhead Hessian.

    Coordinates are the eight unconstrained plate intercepts followed by compound-major,
    dose-major effects.  The Hessian is for the maximized canonical fixed-q objective, so its
    Newton increment solves ``H @ increment = -gradient``.
    """

    if type(design) is not SciPlex3V5Design:
        raise SciPlex3V5ObjectiveError("v5 joint derivatives require the exact design type")
    posterior = _freeze_float(
        posterior_well_means,
        shape=(SCIPLEX3_V5_TRAINING_WELL_COUNT,),
        name="v5 factor posterior well means",
    )
    beta = _freeze_float(
        plate_intercepts,
        shape=(SCIPLEX3_V5_PLATE_COUNT,),
        name="v5 factor plate intercepts",
    )
    effects = _freeze_float(
        delta,
        shape=(SCIPLEX3_V5_COMPOUND_COUNT, SCIPLEX3_V5_DOSE_COUNT),
        name="v5 factor dose effects",
    )
    if bool(np.any(posterior <= 0.0)):
        raise SciPlex3V5ObjectiveError("v5 factor posterior well means must be positive")

    eta = np.asarray(beta[design.training_well_plate_indices], dtype=np.float64).copy()
    action_wells = design.action_well_indices
    eta[action_wells.ravel(order="C")] += effects.ravel(order="C")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        curvature = SCIPLEX3_V5_ACTION_LIKELIHOOD_SCALE * posterior * np.exp(-eta)
    eta_gradient = curvature - SCIPLEX3_V5_ACTION_LIKELIHOOD_SCALE
    if not bool(np.all(np.isfinite(curvature))) or not bool(np.all(np.isfinite(eta_gradient))):
        raise SciPlex3V5ObjectiveError("v5 joint factor derivatives are nonfinite")

    beta_gradient = np.zeros(SCIPLEX3_V5_PLATE_COUNT, dtype=np.float64)
    beta_curvature = np.zeros(SCIPLEX3_V5_PLATE_COUNT, dtype=np.float64)
    np.add.at(beta_gradient, design.training_well_plate_indices, eta_gradient)
    np.add.at(beta_curvature, design.training_well_plate_indices, curvature)
    penalty_gradient = effects / 4.0 + np.einsum(
        "ds,cs->cd",
        _SECOND_DIFFERENCE.T,
        np.einsum("sd,cd->cs", _SECOND_DIFFERENCE, effects),
    )
    delta_gradient = eta_gradient[action_wells] - penalty_gradient

    delta_coordinate_count = SCIPLEX3_V5_COMPOUND_COUNT * SCIPLEX3_V5_DOSE_COUNT
    coordinate_count = SCIPLEX3_V5_PLATE_COUNT + delta_coordinate_count
    hessian = np.zeros((coordinate_count, coordinate_count), dtype=np.float64)
    beta_coordinates = np.arange(SCIPLEX3_V5_PLATE_COUNT)
    hessian[beta_coordinates, beta_coordinates] = -beta_curvature
    action_curvature = curvature[action_wells]
    for compound_index in range(SCIPLEX3_V5_COMPOUND_COUNT):
        start = SCIPLEX3_V5_PLATE_COUNT + compound_index * SCIPLEX3_V5_DOSE_COUNT
        coordinates = np.arange(start, start + SCIPLEX3_V5_DOSE_COUNT)
        hessian[np.ix_(coordinates, coordinates)] = (
            -np.diag(action_curvature[compound_index]) - _DOSE_PENALTY_HESSIAN
        )
        plates = design.training_well_plate_indices[action_wells[compound_index]]
        for dose_index, coordinate in enumerate(coordinates):
            plate_index = int(plates[dose_index])
            cross = -float(action_curvature[compound_index, dose_index])
            hessian[plate_index, coordinate] = cross
            hessian[coordinate, plate_index] = cross
    if not bool(np.all(np.isfinite(hessian))):
        raise SciPlex3V5ObjectiveError("v5 joint factor Hessian is nonfinite")
    return (
        _freeze_float(
            beta_gradient,
            shape=(SCIPLEX3_V5_PLATE_COUNT,),
            name="v5 joint beta gradient",
        ),
        _freeze_float(
            delta_gradient,
            shape=(SCIPLEX3_V5_COMPOUND_COUNT, SCIPLEX3_V5_DOSE_COUNT),
            name="v5 joint delta gradient",
        ),
        _freeze_float(
            hessian,
            shape=(coordinate_count, coordinate_count),
            name="v5 joint arrowhead Hessian",
        ),
    )


@dataclass(frozen=True, slots=True)
class SciPlex3V5AcceptedSubstep:
    """One accepted plate or factor substep on the canonical full-objective scale."""

    kind: Literal["plate-intercept", "factor-newton"]
    sweep: int
    objective_before: float
    objective_after: float
    independent_objective_before: float
    independent_objective_after: float
    parameters_after: SciPlex3V5ActionParameters = field(repr=False)
    parameter_sha256: str
    factor_index: int | None = None
    newton_iteration: int | None = None
    step_scale: float | None = None

    def __post_init__(self) -> None:
        if (
            type(self.sweep) is not int
            or self.sweep <= 0
            or self.kind not in {"plate-intercept", "factor-newton"}
            or not all(
                type(value) is float and math.isfinite(value)
                for value in (
                    self.objective_before,
                    self.objective_after,
                    self.independent_objective_before,
                    self.independent_objective_after,
                )
            )
            or self.objective_after < self.objective_before
            or self.independent_objective_after < self.independent_objective_before
        ):
            raise SciPlex3V5ObjectiveError("v5 accepted substep decreased a full objective")
        if (
            type(self.parameters_after) is not SciPlex3V5ActionParameters
            or type(self.parameter_sha256) is not str
            or self.parameter_sha256 != action_context_parameter_sha256(self.parameters_after)
            or not _objectives_agree(self.objective_after, self.independent_objective_after)
        ):
            raise SciPlex3V5ObjectiveError("v5 accepted substep state witness is invalid")
        if self.kind == "plate-intercept":
            if any(
                value is not None
                for value in (self.factor_index, self.newton_iteration, self.step_scale)
            ):
                raise SciPlex3V5ObjectiveError("v5 plate-intercept witness has Newton metadata")
            return
        if (
            type(self.factor_index) is not int
            or not 0 <= self.factor_index < SCIPLEX3_V5_FACTOR_COUNT
            or type(self.newton_iteration) is not int
            or not 1 <= self.newton_iteration <= SCIPLEX3_V5_DOSE_NEWTON_MAX_STEPS
            or type(self.step_scale) is not float
            or not 0.0 < self.step_scale <= 1.0
        ):
            raise SciPlex3V5ObjectiveError("v5 factor-Newton witness metadata is invalid")


@dataclass(frozen=True, slots=True)
class SciPlex3V5CompleteBlock:
    """One complete plate-plus-dose sweep and its independently checked objective."""

    sweep: int
    objective_before: float
    objective_after: float
    independent_objective_before: float
    independent_objective_after: float
    maximum_absolute_gradient: float

    def __post_init__(self) -> None:
        if (
            type(self.sweep) is not int
            or self.sweep <= 0
            or not all(
                type(value) is float and math.isfinite(value)
                for value in (
                    self.objective_before,
                    self.objective_after,
                    self.independent_objective_before,
                    self.independent_objective_after,
                    self.maximum_absolute_gradient,
                )
            )
            or self.objective_after < self.objective_before
            or self.independent_objective_after < self.independent_objective_before
            or self.maximum_absolute_gradient < 0.0
        ):
            raise SciPlex3V5ObjectiveError("v5 complete M-step block witness is invalid")
        if not _objectives_agree(self.objective_after, self.independent_objective_after):
            raise SciPlex3V5ObjectiveError("v5 complete-block objective evaluators disagree")


@dataclass(frozen=True, slots=True, eq=False)
class SciPlex3V5ActionContextFit:
    """Converged v5 M-step parameters plus monotonicity and stationarity witnesses."""

    parameters: SciPlex3V5ActionParameters
    initial_objective: float
    initial_independent_objective: float
    final_objective: float
    final_independent_objective: float
    accepted_substeps: tuple[SciPlex3V5AcceptedSubstep, ...]
    complete_blocks: tuple[SciPlex3V5CompleteBlock, ...]
    terminal_gradients: SciPlex3V5ObjectiveGradients

    def __post_init__(self) -> None:
        if type(self.parameters) is not SciPlex3V5ActionParameters:
            raise SciPlex3V5ObjectiveError("v5 fit parameters have the wrong exact type")
        if (
            type(self.initial_objective) is not float
            or type(self.initial_independent_objective) is not float
            or type(self.final_objective) is not float
            or type(self.final_independent_objective) is not float
            or not math.isfinite(self.initial_objective)
            or not math.isfinite(self.initial_independent_objective)
            or not math.isfinite(self.final_objective)
            or not math.isfinite(self.final_independent_objective)
            or self.final_objective < self.initial_objective
            or self.final_independent_objective < self.initial_independent_objective
            or not _objectives_agree(self.initial_objective, self.initial_independent_objective)
            or not _objectives_agree(self.final_objective, self.final_independent_objective)
            or not self.complete_blocks
            or type(self.terminal_gradients) is not SciPlex3V5ObjectiveGradients
            or self.terminal_gradients.maximum_absolute > SCIPLEX3_V5_GRADIENT_TOL
        ):
            raise SciPlex3V5ObjectiveError("v5 action/context fit did not pass its terminal gates")
        if (
            any(type(item) is not SciPlex3V5AcceptedSubstep for item in self.accepted_substeps)
            or any(type(item) is not SciPlex3V5CompleteBlock for item in self.complete_blocks)
            or not self.accepted_substeps
            or self.complete_blocks[0].objective_before != self.initial_objective
            or self.complete_blocks[0].independent_objective_before
            != self.initial_independent_objective
            or self.complete_blocks[-1].objective_after != self.final_objective
            or self.complete_blocks[-1].independent_objective_after
            != self.final_independent_objective
        ):
            raise SciPlex3V5ObjectiveError("v5 action/context fit witness chain is invalid")
        canonical = self.initial_objective
        independent = self.initial_independent_objective
        for substep in self.accepted_substeps:
            if (
                substep.objective_before != canonical
                or substep.independent_objective_before != independent
            ):
                raise SciPlex3V5ObjectiveError("v5 accepted substep witness chain is discontinuous")
            canonical = substep.objective_after
            independent = substep.independent_objective_after
        if (
            canonical != self.final_objective
            or independent != self.final_independent_objective
            or action_context_parameter_sha256(self.parameters)
            != action_context_parameter_sha256(self.accepted_substeps[-1].parameters_after)
        ):
            raise SciPlex3V5ObjectiveError(
                "v5 accepted substep chain does not bind the final state"
            )


def _default_initial_parameters(posterior: FloatArray) -> SciPlex3V5ActionParameters:
    alpha = np.log(np.mean(posterior, axis=0))
    return SciPlex3V5ActionParameters(
        alpha=alpha,
        log_rho=np.zeros((SCIPLEX3_V5_PLATE_COUNT, SCIPLEX3_V5_FACTOR_COUNT), dtype=np.float64),
        delta=np.zeros(
            (
                SCIPLEX3_V5_COMPOUND_COUNT,
                SCIPLEX3_V5_DOSE_COUNT,
                SCIPLEX3_V5_FACTOR_COUNT,
            ),
            dtype=np.float64,
        ),
    )


def _exact_plate_intercept_update(
    posterior: FloatArray, delta: FloatArray, design: SciPlex3V5Design
) -> FloatArray:
    delta_by_well = np.zeros_like(posterior)
    delta_by_well[design.action_well_indices.ravel(order="C")] = delta.reshape(
        SCIPLEX3_V5_COMPOUND_COUNT * SCIPLEX3_V5_DOSE_COUNT,
        SCIPLEX3_V5_FACTOR_COUNT,
    )
    adjusted = np.log(posterior) - delta_by_well
    beta = np.empty((SCIPLEX3_V5_PLATE_COUNT, SCIPLEX3_V5_FACTOR_COUNT), dtype=np.float64)
    for plate_index in range(SCIPLEX3_V5_PLATE_COUNT):
        selected = adjusted[design.training_well_plate_indices == plate_index]
        if len(selected) == 0:
            raise SciPlex3V5ObjectiveError("v5 plate-intercept update found an empty plate")
        beta[plate_index] = logsumexp(selected, axis=0) - math.log(len(selected))
    if not bool(np.all(np.isfinite(beta))):
        raise SciPlex3V5ObjectiveError("v5 plate-intercept update is nonfinite")
    return beta


def _objectives_agree(canonical: float, independent: float) -> bool:
    tolerance = 16.0 * _objective_tolerance(max(abs(canonical), abs(independent)))
    return bool(abs(canonical - independent) <= tolerance)


def _full_objective_proposal_nondecreases(
    current: float,
    proposed: float,
    current_independent: float,
    proposed_independent: float,
) -> bool:
    """Accept only representable nondecrease from both fresh full-objective evaluators."""

    return bool(
        math.isfinite(proposed)
        and math.isfinite(proposed_independent)
        and proposed >= current
        and proposed_independent >= current_independent
    )


def _factor_newton_step(
    posterior: FloatArray,
    beta: FloatArray,
    delta: FloatArray,
    design: SciPlex3V5Design,
) -> tuple[FloatArray, FloatArray, float]:
    """Solve one factor's arrowhead Newton system through exact four-dose blocks."""

    eta = np.asarray(beta[design.training_well_plate_indices], dtype=np.float64).copy()
    eta[design.action_well_indices.ravel(order="C")] += delta.ravel(order="C")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        curvature = SCIPLEX3_V5_ACTION_LIKELIHOOD_SCALE * posterior * np.exp(-eta)
    eta_gradient = SCIPLEX3_V5_ACTION_LIKELIHOOD_SCALE - curvature
    if not bool(np.all(np.isfinite(curvature))) or not bool(np.all(np.isfinite(eta_gradient))):
        raise SciPlex3V5ObjectiveError("v5 factor Newton derivatives are nonfinite")

    beta_gradient = np.zeros(SCIPLEX3_V5_PLATE_COUNT, dtype=np.float64)
    beta_curvature = np.zeros(SCIPLEX3_V5_PLATE_COUNT, dtype=np.float64)
    np.add.at(beta_gradient, design.training_well_plate_indices, eta_gradient)
    np.add.at(beta_curvature, design.training_well_plate_indices, curvature)
    action_wells = design.action_well_indices
    action_gradient = eta_gradient[action_wells]
    action_curvature = curvature[action_wells]
    delta_gradient = delta / 4.0 + np.einsum(
        "ds,cs->cd",
        _SECOND_DIFFERENCE.T,
        np.einsum("sd,cd->cs", _SECOND_DIFFERENCE, delta),
    )
    delta_gradient += action_gradient

    schur = np.diag(beta_curvature)
    right = beta_gradient.copy()
    block_hessians: list[FloatArray] = []
    block_crosses: list[FloatArray] = []
    for compound_index in range(SCIPLEX3_V5_COMPOUND_COUNT):
        hessian = np.diag(action_curvature[compound_index]) + _DOSE_PENALTY_HESSIAN
        cross = np.zeros((SCIPLEX3_V5_PLATE_COUNT, SCIPLEX3_V5_DOSE_COUNT), dtype=np.float64)
        plates = design.training_well_plate_indices[action_wells[compound_index]]
        for dose_index in range(SCIPLEX3_V5_DOSE_COUNT):
            cross[int(plates[dose_index]), dose_index] = action_curvature[
                compound_index, dose_index
            ]
        try:
            solved_cross = np.linalg.solve(hessian, cross.T)
            solved_gradient = np.linalg.solve(hessian, delta_gradient[compound_index])
        except np.linalg.LinAlgError as error:
            raise SciPlex3V5ObjectiveError("v5 four-dose Newton block is singular") from error
        schur -= cross @ solved_cross
        right -= cross @ solved_gradient
        block_hessians.append(hessian)
        block_crosses.append(cross)
    try:
        beta_step = np.linalg.solve(schur, right)
    except np.linalg.LinAlgError as error:
        raise SciPlex3V5ObjectiveError(
            "v5 plate-intercept Newton Schur system is singular"
        ) from error
    delta_step = np.empty_like(delta)
    for compound_index, (hessian, cross) in enumerate(
        zip(block_hessians, block_crosses, strict=True)
    ):
        try:
            delta_step[compound_index] = np.linalg.solve(
                hessian,
                delta_gradient[compound_index] - cross.T @ beta_step,
            )
        except np.linalg.LinAlgError as error:  # pragma: no cover - same matrix solved above
            raise SciPlex3V5ObjectiveError(
                "v5 four-dose Newton recovery block is singular"
            ) from error
    maximum_gradient = max(
        float(np.max(np.abs(beta_gradient))),
        float(np.max(np.abs(delta_gradient))),
    )
    return (
        np.asarray(beta_step, dtype=np.float64),
        np.asarray(delta_step, dtype=np.float64),
        maximum_gradient,
    )


def fit_fixed_q_action_context_m_step(
    posterior_well_factor_means: object,
    design: SciPlex3V5Design,
    *,
    initial: SciPlex3V5ActionParameters | None = None,
) -> SciPlex3V5ActionContextFit:
    """Run the deterministic all-well plate-intercept/four-dose monotone v5 M-step."""

    if type(design) is not SciPlex3V5Design:
        raise SciPlex3V5ObjectiveError("v5 M-step requires the exact design type")
    posterior = _validate_posterior_means(posterior_well_factor_means)
    if initial is None:
        parameters = _default_initial_parameters(posterior)
    elif type(initial) is SciPlex3V5ActionParameters:
        parameters = initial
    else:
        raise SciPlex3V5ObjectiveError("v5 M-step initial parameters have the wrong exact type")
    beta = _plate_intercepts(parameters).copy()
    delta = np.asarray(parameters.delta, dtype=np.float64).copy()
    initial_objective = fixed_q_full_elbo_action_context(posterior, parameters, design)
    independent_initial = independent_fixed_q_full_elbo_action_context(
        posterior, parameters, design
    )
    if not _objectives_agree(initial_objective, independent_initial):
        raise SciPlex3V5ObjectiveError("v5 canonical and independent initial objectives disagree")
    running_objective = initial_objective
    running_independent_objective = independent_initial
    substeps: list[SciPlex3V5AcceptedSubstep] = []
    blocks: list[SciPlex3V5CompleteBlock] = []

    block_before = running_objective
    independent_block_before = running_independent_objective
    proposed_beta = _exact_plate_intercept_update(posterior, delta, design)
    proposed_parameters = _parameters_from_plate_intercepts(proposed_beta, delta)
    proposed_objective = fixed_q_full_elbo_action_context(posterior, proposed_parameters, design)
    proposed_independent = independent_fixed_q_full_elbo_action_context(
        posterior, proposed_parameters, design
    )
    if not _objectives_agree(proposed_objective, proposed_independent):
        raise SciPlex3V5ObjectiveError(
            "v5 plate update canonical and independent objectives disagree"
        )
    if _full_objective_proposal_nondecreases(
        running_objective,
        proposed_objective,
        running_independent_objective,
        proposed_independent,
    ):
        substeps.append(
            SciPlex3V5AcceptedSubstep(
                kind="plate-intercept",
                sweep=1,
                objective_before=float(running_objective),
                objective_after=float(proposed_objective),
                independent_objective_before=float(running_independent_objective),
                independent_objective_after=float(proposed_independent),
                parameters_after=proposed_parameters,
                parameter_sha256=action_context_parameter_sha256(proposed_parameters),
            )
        )
        beta = proposed_beta
        running_objective = proposed_objective
        running_independent_objective = proposed_independent

    for factor_index in range(SCIPLEX3_V5_FACTOR_COUNT):
        posterior_factor = np.asarray(posterior[:, factor_index], dtype=np.float64)
        beta_factor = np.asarray(beta[:, factor_index], dtype=np.float64).copy()
        delta_factor = np.asarray(delta[:, :, factor_index], dtype=np.float64).copy()
        for newton_iteration in range(1, SCIPLEX3_V5_DOSE_NEWTON_MAX_STEPS + 1):
            beta_step, delta_step, maximum_gradient = _factor_newton_step(
                posterior_factor, beta_factor, delta_factor, design
            )
            if maximum_gradient <= _INTERNAL_GRADIENT_TOL:
                break
            accepted = False
            step_scale = 1.0
            for _ in range(SCIPLEX3_V5_BACKTRACK_MAX_STEPS + 1):
                proposed_beta_factor = beta_factor - step_scale * beta_step
                proposed_delta_factor = delta_factor - step_scale * delta_step
                proposed_beta = beta.copy()
                proposed_beta[:, factor_index] = proposed_beta_factor
                proposed_delta = delta.copy()
                proposed_delta[:, :, factor_index] = proposed_delta_factor
                proposed_parameters = _parameters_from_plate_intercepts(
                    proposed_beta, proposed_delta
                )
                proposed_full_objective = fixed_q_full_elbo_action_context(
                    posterior, proposed_parameters, design
                )
                proposed_independent = independent_fixed_q_full_elbo_action_context(
                    posterior, proposed_parameters, design
                )
                if not _objectives_agree(proposed_full_objective, proposed_independent):
                    raise SciPlex3V5ObjectiveError(
                        "v5 Newton proposal canonical and independent objectives disagree"
                    )
                if _full_objective_proposal_nondecreases(
                    running_objective,
                    proposed_full_objective,
                    running_independent_objective,
                    proposed_independent,
                ):
                    beta_factor = proposed_beta_factor
                    delta_factor = proposed_delta_factor
                    beta = proposed_beta
                    delta = proposed_delta
                    substeps.append(
                        SciPlex3V5AcceptedSubstep(
                            kind="factor-newton",
                            sweep=1,
                            objective_before=float(running_objective),
                            objective_after=float(proposed_full_objective),
                            independent_objective_before=float(running_independent_objective),
                            independent_objective_after=float(proposed_independent),
                            parameters_after=proposed_parameters,
                            parameter_sha256=action_context_parameter_sha256(proposed_parameters),
                            factor_index=factor_index,
                            newton_iteration=newton_iteration,
                            step_scale=float(step_scale),
                        )
                    )
                    running_objective = proposed_full_objective
                    running_independent_objective = proposed_independent
                    accepted = True
                    break
                step_scale *= 0.5
            if not accepted:
                raise SciPlex3V5ObjectiveError(
                    "v5 factor Newton step failed monotone backtracking above its "
                    f"gradient tolerance (factor={factor_index}, "
                    f"max_gradient={maximum_gradient})"
                )
        else:
            raise SciPlex3V5ObjectiveError(
                f"v5 factor Newton block {factor_index} failed its gradient tolerance"
            )
        beta[:, factor_index] = beta_factor
        delta[:, :, factor_index] = delta_factor

    parameters = _parameters_from_plate_intercepts(beta, delta)
    canonical = fixed_q_full_elbo_action_context(posterior, parameters, design)
    independent = independent_fixed_q_full_elbo_action_context(posterior, parameters, design)
    if not _objectives_agree(canonical, independent):
        raise SciPlex3V5ObjectiveError(
            "v5 canonical and independent complete-block objectives disagree"
        )
    if canonical < block_before:
        raise SciPlex3V5ObjectiveError("v5 complete action/context block decreased the ELBO")
    if canonical != running_objective or independent != running_independent_objective:
        raise SciPlex3V5ObjectiveError(
            "v5 accepted-substep witnesses do not reconcile to the full objective"
        )
    gradients = fixed_q_full_elbo_action_context_gradients(posterior, parameters, design)
    blocks.append(
        SciPlex3V5CompleteBlock(
            sweep=1,
            objective_before=float(block_before),
            objective_after=float(canonical),
            independent_objective_before=float(independent_block_before),
            independent_objective_after=float(independent),
            maximum_absolute_gradient=float(gradients.maximum_absolute),
        )
    )
    return SciPlex3V5ActionContextFit(
        parameters=parameters,
        initial_objective=float(initial_objective),
        initial_independent_objective=float(independent_initial),
        final_objective=float(canonical),
        final_independent_objective=float(independent),
        accepted_substeps=tuple(substeps),
        complete_blocks=tuple(blocks),
        terminal_gradients=gradients,
    )


__all__ = [
    "SCIPLEX3_V5_ACTION_LIKELIHOOD_SCALE",
    "SCIPLEX3_V5_EQUAL_WELL_SCALE",
    "SCIPLEX3_V5_GRADIENT_TOL",
    "SCIPLEX3_V5_OBJECTIVE_VERSION",
    "SciPlex3V5AcceptedSubstep",
    "SciPlex3V5ActionContextFit",
    "SciPlex3V5ActionParameters",
    "SciPlex3V5CompleteBlock",
    "SciPlex3V5Design",
    "SciPlex3V5ObjectiveError",
    "SciPlex3V5ObjectiveGradients",
    "action_context_parameter_sha256",
    "feasible_coordinates_from_log_rho",
    "fit_fixed_q_action_context_m_step",
    "fixed_q_dose_block_gradient_hessian",
    "fixed_q_factor_arrowhead_gradient_hessian",
    "fixed_q_full_elbo_action_context",
    "fixed_q_full_elbo_action_context_gradients",
    "independent_fixed_q_full_elbo_action_context",
    "log_rho_from_feasible_coordinates",
]
