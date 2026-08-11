"""Source-free v5 sampling and request-support primitives for sci-Plex3.

This module deliberately does not load, fit, or inspect a dataset.  A future v5 candidate can
compose these primitives after it has produced immutable action log means, calibrated context
multipliers, factor shapes, and loading rows.  The observable model is the same continuous
Gamma--Poisson admixture used by the retired candidate family, but sampling is exact conditional
on a positive panel rather than bounded positive-panel rejection.

For factor ``k``, integrating ``theta_k ~ Gamma(r, scale=m_k/r)`` out of the factor-total Poisson
count gives ``N_k ~ NegativeBinomial(r, p_k=r/(r+m_k))``.  Its compound-Poisson representation is

``L_k ~ Poisson(-r log(p_k))`` and ``N_k = sum_j LogSeries(1-p_k)``.

Conditioning the superposed cluster count ``sum_k L_k`` to be positive therefore conditions the
whole panel exactly.  The only residual probabilistic failure is the unrepresentable ``int64``
tail.  It is admitted only when a conservative conditional Chernoff bound is at most ``2**-64``
for a complete 512-draw request over every declared action, context, and calibration state.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from types import MappingProxyType
from typing import Final, Literal, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SCIPLEX3_V5_MAX_SAMPLE_COUNT: Final = 512
SCIPLEX3_V5_INT64_EXCLUSIVE_UPPER_BOUND: Final = 1 << 63
SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG: Final = -64.0 * math.log(2.0)
SCIPLEX3_V5_MAX_COMPOUND_POISSON_INTENSITY: Final = 64.0
SCIPLEX3_V5_RNG_ALGORITHM: Final = "numpy-pcg64dxsm-v1"


class SciPlex3SamplingV5Error(ValueError):
    """Raised when v5 sampling or support violates the frozen source-free contract."""


class SciPlex3SamplingV5OverflowError(SciPlex3SamplingV5Error):
    """Raised on the budgeted but unrepresentable signed-int64 count tail."""


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
        raise SciPlex3SamplingV5Error(
            "v5 sampling payload is not canonical-JSON compatible"
        ) from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise SciPlex3SamplingV5Error(f"{name} must be a nonblank trimmed string")
    return value


def _strict_sha256(value: object, *, name: str) -> str:
    text = _strict_text(value, name=name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise SciPlex3SamplingV5Error(f"{name} must be a lowercase SHA-256 digest")
    return text


def _strict_labels(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    labels = tuple(values)
    if not labels or len(set(labels)) != len(labels):
        raise SciPlex3SamplingV5Error(f"{name} must be nonempty and unique")
    for label in labels:
        _strict_text(label, name=name)
    return labels


def _freeze_float_array(
    value: object,
    *,
    ndim: int,
    name: str,
    strictly_positive: bool = False,
    nonnegative: bool = False,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.ndim != ndim or raw.dtype.kind not in {"f", "i", "u"}:
        raise SciPlex3SamplingV5Error(f"{name} has an invalid shape or dtype")
    canonical = np.asarray(raw, dtype="<f8", order="C")
    if not bool(np.all(np.isfinite(canonical))):
        raise SciPlex3SamplingV5Error(f"{name} must be finite")
    if strictly_positive and bool(np.any(canonical <= 0.0)):
        raise SciPlex3SamplingV5Error(f"{name} must be strictly positive")
    if nonnegative and bool(np.any(canonical < 0.0)):
        raise SciPlex3SamplingV5Error(f"{name} must be nonnegative")
    return np.frombuffer(canonical.tobytes(order="C"), dtype="<f8").reshape(canonical.shape)


def _array_sha256(value: NDArray[np.generic]) -> str:
    canonical = np.asarray(value, order="C")
    return _sha256_bytes(canonical.tobytes(order="C"))


_SAMPLING_CONTRACT_MANIFEST: Final[Mapping[str, object]] = MappingProxyType(
    {
        "conditioning": "exact-positive-panel-via-zero-truncated-compound-poisson",
        "count_tail": {
            "exclusive_upper_bound": SCIPLEX3_V5_INT64_EXCLUSIVE_UPPER_BOUND,
            "request_budget": "2^-64",
            "bound": "conditional-negative-binomial-chernoff-half-abscissa",
        },
        "factor_count_marginal": "negative-binomial-gamma-poisson",
        "feature_allocation": "factor-total-multinomial-loading-row",
        "maximum_compound_poisson_intensity": (SCIPLEX3_V5_MAX_COMPOUND_POISSON_INTENSITY),
        "maximum_request_count": SCIPLEX3_V5_MAX_SAMPLE_COUNT,
        "request_support": "exact-request-type-and-global-action-context-calibration-envelope",
        "rng": SCIPLEX3_V5_RNG_ALGORITHM,
        "rng_substreams": {
            "context": "whole-context-model-context-key-seed",
            "row": "model-calibration-target-context-seed-draw-index",
        },
        "schema": "sciplex3-v5-positive-conditioned-sampling-contract",
        "version": "1.0.0",
    }
)
SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256: Final = _sha256_bytes(
    _canonical_json_bytes(dict(_SAMPLING_CONTRACT_MANIFEST))
)


def sampling_contract_manifest() -> dict[str, object]:
    """Return a detached canonical description of the frozen v5 sampling contract."""

    return cast(
        dict[str, object], json.loads(_canonical_json_bytes(dict(_SAMPLING_CONTRACT_MANIFEST)))
    )


def canonical_target_fingerprint(payload: Mapping[str, object]) -> str:
    """Fingerprint a caller-owned, outcome-free target identity for RNG and provenance binding."""

    if not isinstance(payload, Mapping):
        raise SciPlex3SamplingV5Error("v5 sampling target payload must be a mapping")
    return _sha256_bytes(_canonical_json_bytes(dict(payload)))


@dataclass(frozen=True, slots=True)
class V5SamplingTarget:
    """Generic exact target identity supplied by a future v5 candidate adapter."""

    target_fingerprint: str
    action_id: str
    context_key: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_fingerprint",
            _strict_sha256(self.target_fingerprint, name="v5 target fingerprint"),
        )
        _strict_text(self.action_id, name="v5 target action ID")
        _strict_text(self.context_key, name="v5 target context key")


@dataclass(frozen=True, slots=True)
class V5SampleRequest:
    """Seed-bound exact request; the support decision includes its complete sample count."""

    target: V5SamplingTarget
    sample_count: int
    seed: int

    def __post_init__(self) -> None:
        if type(self.target) is not V5SamplingTarget:
            raise SciPlex3SamplingV5Error("v5 request target must have the exact target type")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise SciPlex3SamplingV5Error("v5 sample count must be a positive exact integer")
        if type(self.seed) is not int or not 0 <= self.seed <= np.iinfo(np.uint64).max:
            raise SciPlex3SamplingV5Error("v5 seed must be an unsigned 64-bit exact integer")


@dataclass(frozen=True, slots=True, eq=False)
class V5SamplingParameters:
    """Immutable generic arrays from which a future v5 candidate can sample.

    ``action_log_means`` has shape ``[action, factor]``.  ``context_multipliers`` has shape
    ``[calibration, context, factor]`` and is already transformed under the future candidate's
    accepted context/calibration rule.  This module intentionally does not choose that rule.
    """

    model_artifact_sha256: str
    action_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    calibration_taus: tuple[float, ...]
    active_tau: float
    action_log_means: FloatArray = field(repr=False)
    context_multipliers: FloatArray = field(repr=False)
    factor_shapes: FloatArray = field(repr=False)
    basis: FloatArray = field(repr=False)

    def __post_init__(self) -> None:
        model_sha256 = _strict_sha256(self.model_artifact_sha256, name="v5 model artifact sha256")
        actions = _strict_labels(self.action_ids, name="v5 action IDs")
        contexts = _strict_labels(self.context_ids, name="v5 context IDs")
        taus = tuple(self.calibration_taus)
        if (
            not taus
            or any(type(tau) is not float or not math.isfinite(tau) or tau <= 0.0 for tau in taus)
            or len({tau.hex() for tau in taus}) != len(taus)
        ):
            raise SciPlex3SamplingV5Error(
                "v5 calibration taus must be unique finite positive exact floats"
            )
        if type(self.active_tau) is not float or self.active_tau.hex() not in {
            tau.hex() for tau in taus
        }:
            raise SciPlex3SamplingV5Error("v5 active tau must exactly match the declared grid")

        action_log_means = _freeze_float_array(
            self.action_log_means,
            ndim=2,
            name="v5 action log means",
        )
        multipliers = _freeze_float_array(
            self.context_multipliers,
            ndim=3,
            name="v5 context multipliers",
            strictly_positive=True,
        )
        factor_shapes = _freeze_float_array(
            self.factor_shapes,
            ndim=1,
            name="v5 factor shapes",
            strictly_positive=True,
        )
        basis = _freeze_float_array(
            self.basis,
            ndim=2,
            name="v5 loading basis",
            nonnegative=True,
        )
        factor_count = action_log_means.shape[1]
        if (
            action_log_means.shape != (len(actions), factor_count)
            or multipliers.shape != (len(taus), len(contexts), factor_count)
            or factor_shapes.shape != (len(taus),)
            or basis.shape[0] != factor_count
            or basis.shape[1] == 0
        ):
            raise SciPlex3SamplingV5Error("v5 sampling parameter shapes disagree")
        if not bool(np.allclose(np.sum(basis, axis=1), 1.0, rtol=0.0, atol=5e-13)):
            raise SciPlex3SamplingV5Error("v5 loading rows must sum to one")

        object.__setattr__(self, "model_artifact_sha256", model_sha256)
        object.__setattr__(self, "action_ids", actions)
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "calibration_taus", taus)
        object.__setattr__(self, "action_log_means", action_log_means)
        object.__setattr__(self, "context_multipliers", multipliers)
        object.__setattr__(self, "factor_shapes", factor_shapes)
        object.__setattr__(self, "basis", basis)

    @property
    def active_tau_index(self) -> int:
        """Return the bit-exact active calibration position."""

        active_hex = self.active_tau.hex()
        return next(
            index for index, tau in enumerate(self.calibration_taus) if tau.hex() == active_hex
        )

    @property
    def active_calibration_state_sha256(self) -> str:
        """Bind output provenance to the exact active shape and context tensor."""

        index = self.active_tau_index
        return _sha256_bytes(
            _canonical_json_bytes(
                {
                    "context_multipliers_sha256": _array_sha256(self.context_multipliers[index]),
                    "context_shape": list(self.context_multipliers[index].shape),
                    "factor_shape_hex": float(self.factor_shapes[index]).hex(),
                    "sampling_contract_sha256": SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256,
                    "tau_hex": self.calibration_taus[index].hex(),
                }
            )
        )

    @property
    def parameter_fingerprint(self) -> str:
        """Bind a support certificate to every numerical array and ordered support label.

        The model-artifact digest is deliberately excluded.  It affects RNG and result
        provenance, but it cannot affect the numerical support envelope.  This separation lets a
        candidate certify its numerical parameters before its canonical bytes (and therefore its
        model digest) exist, then rebind that already-certified sampler to the completed artifact.
        """

        return _sha256_bytes(
            _canonical_json_bytes(
                {
                    "action_ids": list(self.action_ids),
                    "action_log_means_sha256": _array_sha256(self.action_log_means),
                    "basis_sha256": _array_sha256(self.basis),
                    "calibration_taus_hex": [tau.hex() for tau in self.calibration_taus],
                    "context_ids": list(self.context_ids),
                    "context_multipliers_sha256": _array_sha256(self.context_multipliers),
                    "factor_shapes_sha256": _array_sha256(self.factor_shapes),
                }
            )
        )


def _log1p_positive_ratio(numerator: float, denominator: float) -> float:
    """Compute ``log1p(numerator / denominator)`` without ratio overflow."""

    ratio = numerator / denominator
    if math.isfinite(ratio):
        return math.log1p(ratio)
    return math.log(numerator) - math.log(denominator) + math.log1p(denominator / numerator)


@dataclass(frozen=True, slots=True)
class ConditionalCountTailBound:
    """One conservative Chernoff witness for a positive-conditioned panel total."""

    exclusive_upper_bound: int
    log_zero_probability: float
    compound_poisson_intensity: float
    chernoff_parameter: float
    conditional_log_upper_bound: float


def conditional_count_tail_bound(
    factor_means: object,
    factor_shape: float,
    *,
    exclusive_upper_bound: int = SCIPLEX3_V5_INT64_EXCLUSIVE_UPPER_BOUND,
) -> ConditionalCountTailBound:
    """Bound ``P(total >= bound | total > 0)`` for independent Gamma--Poisson factors.

    The fixed witness uses half the moment-generating function's positive abscissa.  It is not an
    optimizer, which keeps the certificate conservative, deterministic, and independently
    reproducible with scalar arithmetic.
    """

    means = np.asarray(factor_means, dtype=np.float64)
    if (
        means.ndim != 1
        or means.size == 0
        or not bool(np.all(np.isfinite(means)))
        or bool(np.any(means <= 0.0))
    ):
        raise SciPlex3SamplingV5Error("v5 tail-bound factor means must be finite and positive")
    if type(factor_shape) is not float or not math.isfinite(factor_shape) or factor_shape <= 0.0:
        raise SciPlex3SamplingV5Error("v5 tail-bound factor shape must be a positive exact float")
    if type(exclusive_upper_bound) is not int or exclusive_upper_bound <= 0:
        raise SciPlex3SamplingV5Error("v5 tail-bound threshold must be a positive exact integer")

    log_terms = [_log1p_positive_ratio(float(mean), factor_shape) for mean in means]
    compound_intensity = factor_shape * math.fsum(log_terms)
    if not math.isfinite(compound_intensity) or compound_intensity <= 0.0:
        raise SciPlex3SamplingV5Error("v5 compound-Poisson intensity is not representable")
    log_zero_probability = -compound_intensity
    positive_probability = -math.expm1(log_zero_probability)
    if not 0.0 < positive_probability <= 1.0:
        raise SciPlex3SamplingV5Error("v5 positive-panel normalizer is not representable")
    log_positive_probability = math.log(positive_probability)

    maximum_mean = float(np.max(means))
    abscissa = math.log1p(factor_shape / maximum_mean)
    chernoff_parameter = 0.5 * abscissa
    if not math.isfinite(chernoff_parameter) or chernoff_parameter <= 0.0:
        raise SciPlex3SamplingV5Error("v5 Chernoff parameter is not representable")
    scaled_exponential = math.expm1(chernoff_parameter) / factor_shape
    mgf_arguments = [float(mean) * scaled_exponential for mean in means]
    if any(not 0.0 <= argument < 1.0 for argument in mgf_arguments):
        raise SciPlex3SamplingV5Error("v5 Chernoff witness left the MGF domain")
    cumulant = -factor_shape * math.fsum(math.log1p(-argument) for argument in mgf_arguments)
    raw_log_bound = cumulant - chernoff_parameter * exclusive_upper_bound - log_positive_probability
    conditional_log_upper_bound = min(0.0, raw_log_bound)
    if not math.isfinite(conditional_log_upper_bound):
        raise SciPlex3SamplingV5Error("v5 conditional count-tail bound is not finite")
    return ConditionalCountTailBound(
        exclusive_upper_bound=exclusive_upper_bound,
        log_zero_probability=log_zero_probability,
        compound_poisson_intensity=compound_intensity,
        chernoff_parameter=chernoff_parameter,
        conditional_log_upper_bound=conditional_log_upper_bound,
    )


def request_tail_log_upper_bound(per_draw_conditional_log_bound: float, sample_count: int) -> float:
    """Apply a conservative union bound to one complete request."""

    if (
        type(per_draw_conditional_log_bound) is not float
        or not math.isfinite(per_draw_conditional_log_bound)
        or per_draw_conditional_log_bound > 0.0
    ):
        raise SciPlex3SamplingV5Error("v5 per-draw log bound must be finite and nonpositive")
    if type(sample_count) is not int or sample_count <= 0:
        raise SciPlex3SamplingV5Error("v5 tail-bound request count must be positive")
    return min(0.0, math.log(sample_count) + per_draw_conditional_log_bound)


def _factor_means_for_combination(
    parameters: V5SamplingParameters,
    action_index: int,
    calibration_index: int,
    context_index: int,
) -> FloatArray:
    log_means = parameters.action_log_means[action_index] + np.log(
        parameters.context_multipliers[calibration_index, context_index]
    )
    if not bool(np.all(np.isfinite(log_means))) or bool(
        np.any(log_means > math.log(np.finfo(np.float64).max))
    ):
        raise SciPlex3SamplingV5Error("v5 factor means overflow float64 support")
    means = np.exp(log_means)
    if not bool(np.all(np.isfinite(means))) or bool(np.any(means <= 0.0)):
        raise SciPlex3SamplingV5Error("v5 factor means are not finite and positive")
    return np.asarray(means, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class V5SamplingEnvelopeCertificate:
    """Global action/context/calibration support decision for one immutable parameter set."""

    parameter_fingerprint: str
    combination_count: int
    supported: bool
    rejection_reasons: tuple[str, ...]
    maximum_request_count: int
    request_failure_budget_log: float
    worst_request_tail_log_upper_bound: float
    worst_action_id: str
    worst_context_id: str
    worst_tau_hex: str
    maximum_compound_poisson_intensity: float
    sampling_contract_sha256: str = SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256

    @property
    def fingerprint(self) -> str:
        """Return the canonical certificate identity."""

        return _sha256_bytes(
            _canonical_json_bytes(
                {
                    "combination_count": self.combination_count,
                    "maximum_compound_poisson_intensity": (self.maximum_compound_poisson_intensity),
                    "maximum_request_count": self.maximum_request_count,
                    "parameter_fingerprint": self.parameter_fingerprint,
                    "rejection_reasons": list(self.rejection_reasons),
                    "request_failure_budget_log": self.request_failure_budget_log,
                    "sampling_contract_sha256": self.sampling_contract_sha256,
                    "supported": self.supported,
                    "worst_action_id": self.worst_action_id,
                    "worst_context_id": self.worst_context_id,
                    "worst_request_tail_log_upper_bound": (self.worst_request_tail_log_upper_bound),
                    "worst_tau_hex": self.worst_tau_hex,
                }
            )
        )


def build_sampling_envelope_certificate(
    parameters: V5SamplingParameters,
) -> V5SamplingEnvelopeCertificate:
    """Audit every declared action/context/calibration combination without source access."""

    if type(parameters) is not V5SamplingParameters:
        raise SciPlex3SamplingV5Error("v5 support certification requires exact parameter type")
    reasons: set[str] = set()
    worst_request_bound = -math.inf
    worst_action = parameters.action_ids[0]
    worst_context = parameters.context_ids[0]
    worst_tau = parameters.calibration_taus[0].hex()
    maximum_intensity = 0.0

    for calibration_index, tau in enumerate(parameters.calibration_taus):
        factor_shape = float(parameters.factor_shapes[calibration_index])
        for action_index, action_id in enumerate(parameters.action_ids):
            for context_index, context_id in enumerate(parameters.context_ids):
                combination_bound = 0.0
                try:
                    factor_means = _factor_means_for_combination(
                        parameters,
                        action_index,
                        calibration_index,
                        context_index,
                    )
                    witness = conditional_count_tail_bound(factor_means, factor_shape)
                    maximum_intensity = max(maximum_intensity, witness.compound_poisson_intensity)
                    combination_bound = request_tail_log_upper_bound(
                        witness.conditional_log_upper_bound,
                        SCIPLEX3_V5_MAX_SAMPLE_COUNT,
                    )
                    probabilities = factor_means / (factor_shape + factor_means)
                    if not bool(np.all((probabilities > 0.0) & (probabilities < 1.0))):
                        reasons.add("log-series probabilities leave float64 RNG support")
                    if (
                        witness.compound_poisson_intensity
                        > SCIPLEX3_V5_MAX_COMPOUND_POISSON_INTENSITY
                    ):
                        reasons.add("compound-Poisson intensity exceeds inverse-sampler support")
                    if combination_bound > SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG:
                        reasons.add("conditional int64 request-tail bound exceeds 2^-64")
                except SciPlex3SamplingV5Error:
                    reasons.add("factor parameters lack finite v5 sampling support")
                if combination_bound > worst_request_bound:
                    worst_request_bound = combination_bound
                    worst_action = action_id
                    worst_context = context_id
                    worst_tau = tau.hex()

    if not math.isfinite(worst_request_bound):
        worst_request_bound = 0.0
    return V5SamplingEnvelopeCertificate(
        parameter_fingerprint=parameters.parameter_fingerprint,
        combination_count=(
            len(parameters.action_ids)
            * len(parameters.context_ids)
            * len(parameters.calibration_taus)
        ),
        supported=not reasons,
        rejection_reasons=tuple(sorted(reasons)),
        maximum_request_count=SCIPLEX3_V5_MAX_SAMPLE_COUNT,
        request_failure_budget_log=SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG,
        worst_request_tail_log_upper_bound=worst_request_bound,
        worst_action_id=worst_action,
        worst_context_id=worst_context,
        worst_tau_hex=worst_tau,
        maximum_compound_poisson_intensity=maximum_intensity,
    )


def zero_truncated_poisson_inverse(intensity: float, uniform: float) -> int:
    """Invert the positive Poisson CDF using stable ``expm1`` normalization.

    ``uniform`` must be the exact ``[0, 1)`` variate supplied by the frozen generator.  The
    intensity cap is part of the support certificate and keeps the recurrence finite and numerically
    resolved without a probabilistic retry cap.
    """

    if (
        type(intensity) is not float
        or not math.isfinite(intensity)
        or not 0.0 < intensity <= SCIPLEX3_V5_MAX_COMPOUND_POISSON_INTENSITY
    ):
        raise SciPlex3SamplingV5Error(
            "v5 zero-truncated Poisson intensity is outside frozen inverse support"
        )
    if type(uniform) is not float or not math.isfinite(uniform) or not 0.0 <= uniform < 1.0:
        raise SciPlex3SamplingV5Error("v5 zero-truncated Poisson uniform must lie in [0, 1)")

    probability = intensity / math.expm1(intensity)
    probabilities = [probability]
    cumulative = probability
    count = 1
    while uniform >= cumulative:
        count += 1
        probability *= intensity / count
        if probability <= 0.0 or not math.isfinite(probability):
            raise SciPlex3SamplingV5Error(
                "v5 zero-truncated Poisson inverse lost finite probability mass"
            )
        probabilities.append(probability)
        updated = math.fsum(probabilities)
        if updated <= cumulative and uniform >= updated:
            # A binary64 CDF can stall within one ulp of one for the largest generator
            # uniforms.  Recompute only that boundary case in high-precision scalar arithmetic;
            # this is still deterministic inverse sampling and introduces no retry or tail cap.
            with localcontext() as context:
                context.prec = 80
                decimal_intensity = Decimal.from_float(intensity)
                decimal_uniform = Decimal.from_float(uniform)
                decimal_probability = decimal_intensity / (decimal_intensity.exp() - Decimal(1))
                decimal_cumulative = decimal_probability
                decimal_count = 1
                while decimal_uniform >= decimal_cumulative:
                    decimal_count += 1
                    decimal_probability *= decimal_intensity / Decimal(decimal_count)
                    if decimal_probability <= 0:
                        raise SciPlex3SamplingV5Error(
                            "v5 high-precision Poisson inverse lost probability mass"
                        )
                    decimal_cumulative += decimal_probability
                return decimal_count
        cumulative = updated
    return count


def _context_index(
    parameters: V5SamplingParameters,
    target: V5SamplingTarget,
    seed: int,
) -> int:
    binding = _canonical_json_bytes(
        {
            "context_key": target.context_key,
            "model_artifact_sha256": parameters.model_artifact_sha256,
            "rng_domain": "v5-whole-context-selection",
            "sampling_contract_sha256": SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256,
            "seed": seed,
        }
    )
    derived_seed = int.from_bytes(hashlib.sha256(binding).digest()[:8], "little")
    generator = np.random.Generator(np.random.PCG64DXSM(derived_seed))
    return int(generator.integers(0, len(parameters.context_ids)))


def _row_generator(
    parameters: V5SamplingParameters,
    target: V5SamplingTarget,
    context_id: str,
    seed: int,
    draw_index: int,
) -> np.random.Generator:
    binding = _canonical_json_bytes(
        {
            "action_id": target.action_id,
            "active_calibration_state_sha256": (parameters.active_calibration_state_sha256),
            "context_id": context_id,
            "draw_index": draw_index,
            "model_artifact_sha256": parameters.model_artifact_sha256,
            "rng_domain": "v5-positive-conditioned-row",
            "sampling_contract_sha256": SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256,
            "seed": seed,
            "target_fingerprint": target.target_fingerprint,
        }
    )
    derived_seed = int.from_bytes(hashlib.sha256(binding).digest()[:8], "little")
    return np.random.Generator(np.random.PCG64DXSM(derived_seed))


def _sample_positive_row(
    factor_means: FloatArray,
    factor_shape: float,
    basis: FloatArray,
    generator: np.random.Generator,
) -> IntArray:
    log_terms = np.asarray(
        [_log1p_positive_ratio(float(mean), factor_shape) for mean in factor_means],
        dtype=np.float64,
    )
    intensities = factor_shape * log_terms
    total_intensity = float(math.fsum(float(value) for value in intensities))
    cluster_total = zero_truncated_poisson_inverse(total_intensity, float(generator.random()))
    factor_probabilities = intensities / total_intensity
    factor_probabilities = factor_probabilities / np.sum(factor_probabilities, dtype=np.float64)
    try:
        cluster_counts = generator.multinomial(cluster_total, factor_probabilities)
    except (OverflowError, ValueError) as error:
        raise SciPlex3SamplingV5OverflowError(
            "v5 factor-cluster allocation exceeded frozen RNG support"
        ) from error

    row = np.zeros(basis.shape[1], dtype=np.int64)
    panel_total = 0
    signed_maximum = SCIPLEX3_V5_INT64_EXCLUSIVE_UPPER_BOUND - 1
    for factor_index, raw_cluster_count in enumerate(cluster_counts):
        cluster_count = int(raw_cluster_count)
        if cluster_count == 0:
            continue
        probability = float(
            factor_means[factor_index] / (factor_shape + factor_means[factor_index])
        )
        try:
            cluster_sizes = generator.logseries(probability, size=cluster_count)
        except (OverflowError, ValueError) as error:
            raise SciPlex3SamplingV5OverflowError(
                "v5 logarithmic-series count exceeded frozen RNG support"
            ) from error
        factor_total = sum(int(value) for value in cluster_sizes)
        if (
            factor_total <= 0
            or factor_total >= SCIPLEX3_V5_INT64_EXCLUSIVE_UPPER_BOUND
            or panel_total > signed_maximum - factor_total
        ):
            raise SciPlex3SamplingV5OverflowError(
                "v5 positive-conditioned panel total exceeded signed-int64 support"
            )
        try:
            factor_counts = np.asarray(
                generator.multinomial(factor_total, basis[factor_index]), dtype=np.int64
            )
        except (OverflowError, ValueError) as error:
            raise SciPlex3SamplingV5OverflowError(
                "v5 feature allocation exceeded frozen RNG support"
            ) from error
        if bool(np.any(row > signed_maximum - factor_counts)):
            raise SciPlex3SamplingV5OverflowError("v5 feature count exceeded signed-int64 support")
        row += factor_counts
        panel_total += factor_total
    if panel_total <= 0:
        raise SciPlex3SamplingV5Error(
            "v5 exact positive-conditioning construction produced a zero panel"
        )
    return row


def freeze_positive_int64_samples(value: object) -> IntArray:
    """Validate positive panel totals with Python integers, avoiding int64 reduction overflow."""

    raw = np.asarray(value)
    if (
        raw.ndim != 2
        or raw.shape[0] == 0
        or raw.shape[1] == 0
        or raw.dtype.kind
        not in {
            "i",
            "u",
        }
    ):
        raise SciPlex3SamplingV5Error("v5 raw-count samples have an invalid shape or dtype")
    signed_maximum = SCIPLEX3_V5_INT64_EXCLUSIVE_UPPER_BOUND - 1
    for row in raw:
        total = 0
        for raw_value in row:
            count = int(raw_value)
            if count < 0 or count > signed_maximum:
                raise SciPlex3SamplingV5OverflowError(
                    "v5 raw-count feature is outside signed-int64 support"
                )
            total += count
            if total > signed_maximum:
                raise SciPlex3SamplingV5OverflowError(
                    "v5 raw-count panel total is outside signed-int64 support"
                )
        if total <= 0:
            raise SciPlex3SamplingV5Error("v5 raw-count sample panels must be positive")
    canonical = np.asarray(raw, dtype="<i8", order="C")
    return np.frombuffer(canonical.tobytes(order="C"), dtype="<i8").reshape(canonical.shape)


@dataclass(frozen=True, slots=True, eq=False)
class V5RawCountSamples:
    """Immutable positive raw counts with exact model/calibration/contract provenance."""

    model_artifact_sha256: str
    calibration_state_sha256: str
    sampling_contract_sha256: str
    target_fingerprint: str
    action_id: str
    context_id: str
    seed: int
    samples: IntArray
    rng_algorithm: Literal["numpy-pcg64dxsm-v1"] = SCIPLEX3_V5_RNG_ALGORITHM

    def __post_init__(self) -> None:
        for value, name in (
            (self.model_artifact_sha256, "v5 sample model artifact sha256"),
            (self.calibration_state_sha256, "v5 sample calibration state sha256"),
            (self.sampling_contract_sha256, "v5 sample contract sha256"),
            (self.target_fingerprint, "v5 sample target fingerprint"),
        ):
            _strict_sha256(value, name=name)
        if self.sampling_contract_sha256 != SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256:
            raise SciPlex3SamplingV5Error("v5 sample contract provenance drifted")
        _strict_text(self.action_id, name="v5 sample action ID")
        _strict_text(self.context_id, name="v5 sample context ID")
        if type(self.seed) is not int or not 0 <= self.seed <= np.iinfo(np.uint64).max:
            raise SciPlex3SamplingV5Error("v5 sample seed is invalid")
        if self.rng_algorithm != SCIPLEX3_V5_RNG_ALGORITHM:
            raise SciPlex3SamplingV5Error("v5 sample RNG provenance drifted")
        object.__setattr__(self, "samples", freeze_positive_int64_samples(self.samples))


@dataclass(frozen=True, slots=True)
class V5RequestSupportDecision:
    """Auditable request-level view over the global envelope certificate."""

    supported: bool
    reason: str
    selected_context_id: str | None
    envelope_certificate_sha256: str
    requested_sample_count: int | None
    maximum_sample_count: int = SCIPLEX3_V5_MAX_SAMPLE_COUNT


@dataclass(frozen=True, slots=True)
class V5PositiveConditionedSampler:
    """Composable exact-positive v5 sampler with global fail-closed support."""

    parameters: V5SamplingParameters
    envelope_certificate: V5SamplingEnvelopeCertificate = field(init=False)

    def __post_init__(self) -> None:
        if type(self.parameters) is not V5SamplingParameters:
            raise SciPlex3SamplingV5Error("v5 sampler requires exact parameter type")
        object.__setattr__(
            self,
            "envelope_certificate",
            build_sampling_envelope_certificate(self.parameters),
        )

    def with_model_artifact_sha256(
        self, model_artifact_sha256: str
    ) -> V5PositiveConditionedSampler:
        """Rebind provenance without recomputing the unchanged numerical certificate.

        This is the acyclic construction seam for content-addressed candidates: construct and
        certify once with a placeholder digest, compute canonical model bytes using only the
        certificate's numerical fields, and finally bind RNG/output provenance to the resulting
        model digest.  All support-relevant arrays and labels are reused from this exact sampler.
        """

        rebound_parameters = V5SamplingParameters(
            model_artifact_sha256=model_artifact_sha256,
            action_ids=self.parameters.action_ids,
            context_ids=self.parameters.context_ids,
            calibration_taus=self.parameters.calibration_taus,
            active_tau=self.parameters.active_tau,
            action_log_means=self.parameters.action_log_means,
            context_multipliers=self.parameters.context_multipliers,
            factor_shapes=self.parameters.factor_shapes,
            basis=self.parameters.basis,
        )
        if (
            rebound_parameters.parameter_fingerprint
            != self.envelope_certificate.parameter_fingerprint
        ):
            raise SciPlex3SamplingV5Error(
                "v5 model-provenance rebind changed the certified numerical parameters"
            )
        rebound = object.__new__(type(self))
        object.__setattr__(rebound, "parameters", rebound_parameters)
        object.__setattr__(rebound, "envelope_certificate", self.envelope_certificate)
        return rebound

    def support_decision(self, request: object) -> V5RequestSupportDecision:
        """Evaluate exact request type, count, action, and the complete numeric envelope."""

        certificate_sha256 = self.envelope_certificate.fingerprint
        if type(request) is not V5SampleRequest:
            return V5RequestSupportDecision(
                False,
                "request must have the exact v5 request type",
                None,
                certificate_sha256,
                None,
            )
        exact_request = request
        selected_index = _context_index(self.parameters, exact_request.target, exact_request.seed)
        selected_context = self.parameters.context_ids[selected_index]
        if exact_request.sample_count > SCIPLEX3_V5_MAX_SAMPLE_COUNT:
            return V5RequestSupportDecision(
                False,
                "request sample count exceeds the frozen maximum",
                selected_context,
                certificate_sha256,
                exact_request.sample_count,
            )
        if exact_request.target.action_id not in self.parameters.action_ids:
            return V5RequestSupportDecision(
                False,
                "request action is outside the immutable action support",
                selected_context,
                certificate_sha256,
                exact_request.sample_count,
            )
        if not self.envelope_certificate.supported:
            return V5RequestSupportDecision(
                False,
                "global action/context/calibration sampling envelope is unsupported",
                selected_context,
                certificate_sha256,
                exact_request.sample_count,
            )
        return V5RequestSupportDecision(
            True,
            "supported",
            selected_context,
            certificate_sha256,
            exact_request.sample_count,
        )

    def supports(self, request: object) -> bool:
        """Return request-level support; target-only queries intentionally fail."""

        return self.support_decision(request).supported

    def sample(self, request: object) -> V5RawCountSamples:
        """Generate prefix-stable exact-positive raw-count rows for one supported request."""

        decision = self.support_decision(request)
        if not decision.supported:
            raise SciPlex3SamplingV5Error(f"unsupported v5 sampling request: {decision.reason}")
        exact_request = cast(V5SampleRequest, request)
        context_id = cast(str, decision.selected_context_id)
        action_index = self.parameters.action_ids.index(exact_request.target.action_id)
        context_index = self.parameters.context_ids.index(context_id)
        calibration_index = self.parameters.active_tau_index
        factor_shape = float(self.parameters.factor_shapes[calibration_index])
        factor_means = _factor_means_for_combination(
            self.parameters,
            action_index,
            calibration_index,
            context_index,
        )
        samples = np.empty(
            (exact_request.sample_count, self.parameters.basis.shape[1]), dtype=np.int64
        )
        for draw_index in range(exact_request.sample_count):
            samples[draw_index] = _sample_positive_row(
                factor_means,
                factor_shape,
                self.parameters.basis,
                _row_generator(
                    self.parameters,
                    exact_request.target,
                    context_id,
                    exact_request.seed,
                    draw_index,
                ),
            )
        return V5RawCountSamples(
            model_artifact_sha256=self.parameters.model_artifact_sha256,
            calibration_state_sha256=(self.parameters.active_calibration_state_sha256),
            sampling_contract_sha256=SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256,
            target_fingerprint=exact_request.target.target_fingerprint,
            action_id=exact_request.target.action_id,
            context_id=context_id,
            seed=exact_request.seed,
            samples=samples,
        )


__all__ = [
    "SCIPLEX3_V5_INT64_EXCLUSIVE_UPPER_BOUND",
    "SCIPLEX3_V5_MAX_COMPOUND_POISSON_INTENSITY",
    "SCIPLEX3_V5_MAX_SAMPLE_COUNT",
    "SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG",
    "SCIPLEX3_V5_RNG_ALGORITHM",
    "SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256",
    "ConditionalCountTailBound",
    "SciPlex3SamplingV5Error",
    "SciPlex3SamplingV5OverflowError",
    "V5PositiveConditionedSampler",
    "V5RawCountSamples",
    "V5RequestSupportDecision",
    "V5SampleRequest",
    "V5SamplingEnvelopeCertificate",
    "V5SamplingParameters",
    "V5SamplingTarget",
    "build_sampling_envelope_certificate",
    "canonical_target_fingerprint",
    "conditional_count_tail_bound",
    "freeze_positive_int64_samples",
    "request_tail_log_upper_bound",
    "sampling_contract_manifest",
    "zero_truncated_poisson_inverse",
]
