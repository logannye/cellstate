from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.stats import nbinom

import cellstate.evaluation.sciplex3_sampling_v5 as sampling_module
from cellstate.evaluation.sciplex3_sampling_v5 import (
    SCIPLEX3_V5_INT64_EXCLUSIVE_UPPER_BOUND,
    SCIPLEX3_V5_MAX_SAMPLE_COUNT,
    SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG,
    SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256,
    SciPlex3SamplingV5Error,
    SciPlex3SamplingV5OverflowError,
    V5PositiveConditionedSampler,
    V5RawCountSamples,
    V5SampleRequest,
    V5SamplingParameters,
    V5SamplingTarget,
    build_sampling_envelope_certificate,
    canonical_target_fingerprint,
    conditional_count_tail_bound,
    freeze_positive_int64_samples,
    request_tail_log_upper_bound,
    sampling_contract_manifest,
    zero_truncated_poisson_inverse,
)


def _parameters(
    *,
    model_sha256: str = "a" * 64,
    active_tau: float = 1.0,
) -> V5SamplingParameters:
    return V5SamplingParameters(
        model_artifact_sha256=model_sha256,
        action_ids=("control", "drug-10"),
        context_ids=("context-a", "context-b"),
        calibration_taus=(0.8, 1.0, 1.2),
        active_tau=active_tau,
        action_log_means=np.log(np.asarray(((1.0, 0.7), (1.5, 0.5)))),
        context_multipliers=np.asarray(
            (
                ((1.0, 1.0), (0.8, 1.2)),
                ((1.0, 1.0), (0.9, 1.1)),
                ((1.0, 1.0), (0.7, 1.3)),
            )
        ),
        factor_shapes=np.asarray((0.2, 0.1, 0.07)),
        basis=np.asarray(((0.6, 0.3, 0.1), (0.1, 0.3, 0.6))),
    )


def _target(*, action_id: str = "drug-10", context_key: str = "future-plate") -> V5SamplingTarget:
    return V5SamplingTarget(
        target_fingerprint=canonical_target_fingerprint(
            {
                "action_id": action_id,
                "case_id": "case-a",
                "context_key": context_key,
                "partition_id": "future-source-free-fixture",
                "well_id": "well-a",
            }
        ),
        action_id=action_id,
        context_key=context_key,
    )


def test_contract_is_content_addressed_exact_positive_and_request_bounded() -> None:
    first = sampling_contract_manifest()
    second = sampling_contract_manifest()

    assert first == second
    assert first is not second
    assert first["conditioning"] == "exact-positive-panel-via-zero-truncated-compound-poisson"
    assert first["maximum_request_count"] == 512
    assert first["count_tail"]["request_budget"] == "2^-64"  # type: ignore[index]
    assert canonical_target_fingerprint(first) == canonical_target_fingerprint(second)
    assert len(SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256) == 64


def test_parameters_are_deeply_immutable_and_calibration_provenance_is_exact() -> None:
    parameters = _parameters()
    same = _parameters()
    other_model = _parameters(model_sha256="b" * 64)
    other_tau = _parameters(active_tau=0.8)

    assert parameters.parameter_fingerprint == same.parameter_fingerprint
    assert parameters.parameter_fingerprint == other_model.parameter_fingerprint
    assert parameters.active_calibration_state_sha256 == same.active_calibration_state_sha256
    assert parameters.active_calibration_state_sha256 != other_tau.active_calibration_state_sha256
    assert not parameters.action_log_means.flags.writeable
    assert not parameters.context_multipliers.flags.writeable
    assert not parameters.factor_shapes.flags.writeable
    assert not parameters.basis.flags.writeable
    with pytest.raises(ValueError):
        parameters.basis[0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        parameters.active_tau = 0.8  # type: ignore[misc]


def test_chernoff_witness_conservatively_bounds_an_independent_small_tail() -> None:
    means = np.asarray((1.5, 0.7))
    factor_shape = 0.4
    threshold = 8
    witness = conditional_count_tail_bound(
        means,
        factor_shape,
        exclusive_upper_bound=threshold,
    )

    support = np.arange(threshold)
    first = nbinom.pmf(
        support,
        factor_shape,
        factor_shape / (factor_shape + means[0]),
    )
    second = nbinom.pmf(
        support,
        factor_shape,
        factor_shape / (factor_shape + means[1]),
    )
    convolution = np.convolve(first, second)
    unconditioned_below = math.fsum(float(value) for value in convolution[:threshold])
    zero_probability = math.prod(
        (factor_shape / (factor_shape + float(mean))) ** factor_shape for mean in means
    )
    exact_conditional_tail = (1.0 - unconditioned_below) / (1.0 - zero_probability)

    assert 0.0 < exact_conditional_tail < 1.0
    assert exact_conditional_tail <= math.exp(witness.conditional_log_upper_bound)
    assert witness.log_zero_probability == pytest.approx(math.log(zero_probability))
    assert witness.compound_poisson_intensity == pytest.approx(-math.log(zero_probability))
    assert request_tail_log_upper_bound(witness.conditional_log_upper_bound, 5) == pytest.approx(
        min(0.0, math.log(5) + witness.conditional_log_upper_bound)
    )


def test_zero_truncated_poisson_inverse_matches_exact_probability_intervals() -> None:
    intensity = 0.7
    probability = intensity / math.expm1(intensity)
    lower = 0.0
    cumulative = probability
    for count in range(1, 10):
        uniform = (lower + cumulative) / 2.0
        assert zero_truncated_poisson_inverse(intensity, uniform) == count
        lower = cumulative
        probability *= intensity / (count + 1)
        cumulative = math.fsum((cumulative, probability))

    largest_uniform = float(np.nextafter(1.0, 0.0))
    for value in (1e-12, 0.1, 1.0, 10.0, 64.0):
        assert zero_truncated_poisson_inverse(float(value), largest_uniform) >= 1
    with pytest.raises(SciPlex3SamplingV5Error, match="intensity"):
        zero_truncated_poisson_inverse(0.0, 0.5)
    with pytest.raises(SciPlex3SamplingV5Error, match="uniform"):
        zero_truncated_poisson_inverse(1.0, 1.0)


def test_certificate_checks_every_action_context_and_calibration_state() -> None:
    parameters = _parameters()
    certificate = build_sampling_envelope_certificate(parameters)

    assert certificate.supported
    assert certificate.rejection_reasons == ()
    assert certificate.combination_count == 2 * 2 * 3
    assert certificate.maximum_request_count == SCIPLEX3_V5_MAX_SAMPLE_COUNT
    assert certificate.request_failure_budget_log == SCIPLEX3_V5_REQUEST_FAILURE_BUDGET_LOG
    assert certificate.worst_request_tail_log_upper_bound <= certificate.request_failure_budget_log
    assert certificate.maximum_compound_poisson_intensity > 0.0
    assert len(certificate.fingerprint) == 64


def test_request_support_is_exact_type_count_and_action_scoped_before_allocation() -> None:
    sampler = V5PositiveConditionedSampler(_parameters())
    target = _target()

    assert not sampler.supports(target)
    assert sampler.supports(V5SampleRequest(target, 1, 0))
    assert sampler.supports(V5SampleRequest(target, 512, 0))
    oversized = V5SampleRequest(target, 513, 0)
    assert not sampler.supports(oversized)
    with pytest.raises(SciPlex3SamplingV5Error, match="sample count"):
        sampler.sample(oversized)

    unsupported_action = _target(action_id="unknown")
    request = V5SampleRequest(unsupported_action, 1, 0)
    assert not sampler.supports(request)
    with pytest.raises(SciPlex3SamplingV5Error, match="action"):
        sampler.sample(request)


def test_sampling_is_positive_deterministic_prefix_stable_and_fully_provenanced() -> None:
    parameters = _parameters()
    sampler = V5PositiveConditionedSampler(parameters)
    target = _target()
    short_request = V5SampleRequest(target, 12, 20260811)
    long_request = V5SampleRequest(target, 31, 20260811)

    short = sampler.sample(short_request)
    repeated = sampler.sample(short_request)
    long = sampler.sample(long_request)

    assert np.array_equal(short.samples, repeated.samples)
    assert np.array_equal(short.samples, long.samples[:12])
    assert all(sum(int(value) for value in row) > 0 for row in long.samples)
    assert not short.samples.flags.writeable
    assert short.model_artifact_sha256 == parameters.model_artifact_sha256
    assert short.calibration_state_sha256 == parameters.active_calibration_state_sha256
    assert short.sampling_contract_sha256 == SCIPLEX3_V5_SAMPLING_CONTRACT_SHA256
    assert short.target_fingerprint == target.target_fingerprint
    assert short.action_id == target.action_id
    assert short.context_id in parameters.context_ids


def test_one_factor_observable_matches_positive_conditioned_negative_binomial_mean() -> None:
    mean = 1.5
    factor_shape = 0.7
    parameters = V5SamplingParameters(
        model_artifact_sha256="b" * 64,
        action_ids=("one-action",),
        context_ids=("one-context",),
        calibration_taus=(1.0,),
        active_tau=1.0,
        action_log_means=np.asarray(((math.log(mean),),)),
        context_multipliers=np.ones((1, 1, 1)),
        factor_shapes=np.asarray((factor_shape,)),
        basis=np.ones((1, 1)),
    )
    sampler = V5PositiveConditionedSampler(parameters)
    target = V5SamplingTarget(
        canonical_target_fingerprint({"case": "one-factor"}),
        "one-action",
        "future-context",
    )
    observed = sampler.sample(V5SampleRequest(target, 512, 91)).samples[:, 0]
    zero_probability = (factor_shape / (factor_shape + mean)) ** factor_shape
    expected_conditional_mean = mean / (1.0 - zero_probability)

    assert bool(np.all(observed > 0))
    assert float(np.mean(observed)) == pytest.approx(
        expected_conditional_mean,
        rel=0.12,
        abs=0.0,
    )


def test_safe_selected_context_cannot_bypass_unsafe_global_context_or_tail_budget() -> None:
    parameters = V5SamplingParameters(
        model_artifact_sha256="c" * 64,
        action_ids=("action",),
        context_ids=("safe", "unsafe"),
        calibration_taus=(1.0,),
        active_tau=1.0,
        action_log_means=np.zeros((1, 1)),
        context_multipliers=np.asarray((((1.0,), (8e18,)),)),
        factor_shapes=np.asarray((1e20,)),
        basis=np.ones((1, 1)),
    )
    sampler = V5PositiveConditionedSampler(parameters)
    certificate = sampler.envelope_certificate

    assert not certificate.supported
    assert "conditional int64 request-tail bound exceeds 2^-64" in (certificate.rejection_reasons)
    assert "compound-Poisson intensity exceeds inverse-sampler support" in (
        certificate.rejection_reasons
    )

    target = V5SamplingTarget(
        canonical_target_fingerprint({"case": "safe-seed"}),
        "action",
        "future-context",
    )
    safe_request = None
    for seed in range(100):
        request = V5SampleRequest(target, 1, seed)
        if sampler.support_decision(request).selected_context_id == "safe":
            safe_request = request
            break
    assert safe_request is not None
    decision = sampler.support_decision(safe_request)
    assert decision.selected_context_id == "safe"
    assert not decision.supported
    assert "global" in decision.reason
    with pytest.raises(SciPlex3SamplingV5Error, match="global"):
        sampler.sample(safe_request)


def test_positive_validation_uses_overflow_safe_python_integer_totals() -> None:
    signed_maximum = SCIPLEX3_V5_INT64_EXCLUSIVE_UPPER_BOUND - 1
    valid = freeze_positive_int64_samples(np.asarray(((signed_maximum, 0),), dtype=np.int64))
    assert not valid.flags.writeable
    assert valid[0, 0] == signed_maximum

    with pytest.raises(SciPlex3SamplingV5OverflowError, match="panel total"):
        freeze_positive_int64_samples(np.asarray(((signed_maximum, 1),), dtype=np.int64))
    with pytest.raises(SciPlex3SamplingV5Error, match="positive"):
        freeze_positive_int64_samples(np.zeros((1, 2), dtype=np.int64))
    with pytest.raises(SciPlex3SamplingV5OverflowError, match="feature"):
        freeze_positive_int64_samples(
            np.asarray(((SCIPLEX3_V5_INT64_EXCLUSIVE_UPPER_BOUND,),), dtype=np.uint64)
        )


def test_result_provenance_rejects_contract_drift_and_changes_with_model_or_tau() -> None:
    request = V5SampleRequest(_target(), 3, 7)
    sampler = V5PositiveConditionedSampler(_parameters())
    rebound_sampler = sampler.with_model_artifact_sha256("d" * 64)
    base = sampler.sample(request)
    other_model = rebound_sampler.sample(request)
    other_tau = V5PositiveConditionedSampler(_parameters(active_tau=0.8)).sample(request)

    assert rebound_sampler is not sampler
    assert rebound_sampler.envelope_certificate is sampler.envelope_certificate
    assert (
        rebound_sampler.parameters.parameter_fingerprint == sampler.parameters.parameter_fingerprint
    )
    assert base.model_artifact_sha256 != other_model.model_artifact_sha256
    assert not np.array_equal(base.samples, other_model.samples)
    assert base.calibration_state_sha256 != other_tau.calibration_state_sha256
    assert not np.array_equal(base.samples, other_tau.samples)
    with pytest.raises(SciPlex3SamplingV5Error, match="contract provenance"):
        V5RawCountSamples(
            model_artifact_sha256=base.model_artifact_sha256,
            calibration_state_sha256=base.calibration_state_sha256,
            sampling_contract_sha256="f" * 64,
            target_fingerprint=base.target_fingerprint,
            action_id=base.action_id,
            context_id=base.context_id,
            seed=base.seed,
            samples=base.samples,
        )


def test_parameter_and_request_boundaries_fail_closed() -> None:
    payload = {
        "model_artifact_sha256": "a" * 64,
        "action_ids": ("action",),
        "context_ids": ("context",),
        "calibration_taus": (1.0,),
        "active_tau": 1.0,
        "action_log_means": np.zeros((1, 1)),
        "context_multipliers": np.ones((1, 1, 1)),
        "factor_shapes": np.ones(1),
        "basis": np.ones((1, 1)),
    }
    with pytest.raises(SciPlex3SamplingV5Error, match="active tau"):
        V5SamplingParameters(**{**payload, "active_tau": 1.1})  # type: ignore[arg-type]
    with pytest.raises(SciPlex3SamplingV5Error, match="sum to one"):
        V5SamplingParameters(**{**payload, "basis": np.asarray(((0.5,),))})  # type: ignore[arg-type]
    with pytest.raises(SciPlex3SamplingV5Error, match="positive"):
        V5SampleRequest(_target(), 0, 0)
    with pytest.raises(SciPlex3SamplingV5Error, match="unsigned"):
        V5SampleRequest(_target(), 1, -1)


def test_identity_array_and_provenance_boundaries_fail_closed() -> None:
    with pytest.raises(SciPlex3SamplingV5Error, match="canonical-JSON"):
        canonical_target_fingerprint({"nonfinite": math.nan})
    with pytest.raises(SciPlex3SamplingV5Error, match="mapping"):
        canonical_target_fingerprint([])  # type: ignore[arg-type]
    with pytest.raises(SciPlex3SamplingV5Error, match="trimmed"):
        V5SamplingTarget("a" * 64, " action", "context")
    with pytest.raises(SciPlex3SamplingV5Error, match="SHA-256"):
        V5SamplingTarget("A" * 64, "action", "context")
    with pytest.raises(SciPlex3SamplingV5Error, match="exact target type"):
        V5SampleRequest(object(), 1, 0)  # type: ignore[arg-type]

    payload = {
        "model_artifact_sha256": "a" * 64,
        "action_ids": ("action",),
        "context_ids": ("context",),
        "calibration_taus": (1.0,),
        "active_tau": 1.0,
        "action_log_means": np.zeros((1, 1)),
        "context_multipliers": np.ones((1, 1, 1)),
        "factor_shapes": np.ones(1),
        "basis": np.ones((1, 1)),
    }
    invalid_payloads = (
        ({"action_log_means": np.asarray((("not-a-number",),))}, "shape or dtype"),
        ({"action_log_means": np.asarray(((math.nan,),))}, "finite"),
        ({"context_multipliers": np.zeros((1, 1, 1))}, "strictly positive"),
        ({"basis": np.asarray(((-1.0, 2.0),))}, "nonnegative"),
        ({"calibration_taus": (1.0, 1.0)}, "unique finite positive"),
        ({"context_multipliers": np.ones((1, 1, 2))}, "shapes disagree"),
    )
    for replacement, match in invalid_payloads:
        with pytest.raises(SciPlex3SamplingV5Error, match=match):
            V5SamplingParameters(**{**payload, **replacement})  # type: ignore[arg-type]

    with pytest.raises(SciPlex3SamplingV5Error, match="invalid shape or dtype"):
        freeze_positive_int64_samples(np.ones(2, dtype=np.int64))
    with pytest.raises(SciPlex3SamplingV5Error, match="exact parameter type"):
        V5PositiveConditionedSampler(object())  # type: ignore[arg-type]
    with pytest.raises(SciPlex3SamplingV5Error, match="exact parameter type"):
        build_sampling_envelope_certificate(object())  # type: ignore[arg-type]

    valid = V5PositiveConditionedSampler(_parameters()).sample(V5SampleRequest(_target(), 1, 3))
    result_payload = {
        "model_artifact_sha256": valid.model_artifact_sha256,
        "calibration_state_sha256": valid.calibration_state_sha256,
        "sampling_contract_sha256": valid.sampling_contract_sha256,
        "target_fingerprint": valid.target_fingerprint,
        "action_id": valid.action_id,
        "context_id": valid.context_id,
        "seed": valid.seed,
        "samples": valid.samples,
    }
    with pytest.raises(SciPlex3SamplingV5Error, match="seed"):
        V5RawCountSamples(**{**result_payload, "seed": -1})  # type: ignore[arg-type]
    with pytest.raises(SciPlex3SamplingV5Error, match="RNG provenance"):
        V5RawCountSamples(
            **result_payload,  # type: ignore[arg-type]
            rng_algorithm="different-rng",  # type: ignore[arg-type]
        )


def test_tail_bound_extremes_and_unrepresentable_envelopes_fail_closed() -> None:
    assert math.isfinite(
        sampling_module._log1p_positive_ratio(
            float(np.finfo(np.float64).max),
            float(np.finfo(np.float64).tiny),
        )
    )
    with pytest.raises(SciPlex3SamplingV5Error, match="factor means"):
        conditional_count_tail_bound(np.asarray((0.0,)), 1.0)
    with pytest.raises(SciPlex3SamplingV5Error, match="factor shape"):
        conditional_count_tail_bound(np.asarray((1.0,)), 1)  # type: ignore[arg-type]
    with pytest.raises(SciPlex3SamplingV5Error, match="threshold"):
        conditional_count_tail_bound(np.asarray((1.0,)), 1.0, exclusive_upper_bound=0)
    with pytest.raises(SciPlex3SamplingV5Error, match="intensity"):
        conditional_count_tail_bound(
            np.asarray((float(np.nextafter(0.0, 1.0)),)),
            float(np.finfo(np.float64).max),
        )
    with pytest.raises(SciPlex3SamplingV5Error, match="Chernoff parameter"):
        conditional_count_tail_bound(
            np.asarray((float(np.finfo(np.float64).max),)),
            float(np.nextafter(0.0, 1.0)),
        )
    with pytest.raises(SciPlex3SamplingV5Error, match="per-draw"):
        request_tail_log_upper_bound(0.1, 1)
    with pytest.raises(SciPlex3SamplingV5Error, match="request count"):
        request_tail_log_upper_bound(-1.0, 0)

    base = {
        "model_artifact_sha256": "a" * 64,
        "action_ids": ("action",),
        "context_ids": ("context",),
        "calibration_taus": (1.0,),
        "active_tau": 1.0,
        "context_multipliers": np.ones((1, 1, 1)),
        "factor_shapes": np.ones(1),
        "basis": np.ones((1, 1)),
    }
    for log_mean in (math.log(float(np.finfo(np.float64).max)) + 1.0, -1_000.0):
        parameters = V5SamplingParameters(
            **base,  # type: ignore[arg-type]
            action_log_means=np.asarray(((log_mean,),)),
        )
        certificate = build_sampling_envelope_certificate(parameters)
        assert not certificate.supported
        assert certificate.rejection_reasons == (
            "factor parameters lack finite v5 sampling support",
        )


class _AdversarialGenerator:
    def __init__(self, failure: str) -> None:
        self.failure = failure
        self.multinomial_calls = 0

    def random(self) -> float:
        return 0.0

    def multinomial(self, count: int, probabilities: object) -> np.ndarray:
        del probabilities
        self.multinomial_calls += 1
        if self.failure == "factor-allocation" and self.multinomial_calls == 1:
            raise ValueError("synthetic factor allocation failure")
        if self.failure == "feature-allocation" and self.multinomial_calls == 2:
            raise ValueError("synthetic feature allocation failure")
        if self.multinomial_calls == 1:
            if self.failure == "zero-panel":
                return np.asarray((0,), dtype=np.int64)
            return np.asarray((count,), dtype=np.int64)
        return np.asarray((count,), dtype=np.int64)

    def logseries(self, probability: float, *, size: int) -> np.ndarray:
        del probability
        if self.failure == "log-series":
            raise ValueError("synthetic log-series failure")
        if self.failure == "panel-overflow":
            return np.full(size, SCIPLEX3_V5_INT64_EXCLUSIVE_UPPER_BOUND, dtype=np.uint64)
        return np.ones(size, dtype=np.int64)


@pytest.mark.parametrize(
    ("failure", "exception", "match"),
    (
        ("factor-allocation", SciPlex3SamplingV5OverflowError, "factor-cluster"),
        ("log-series", SciPlex3SamplingV5OverflowError, "logarithmic-series"),
        ("panel-overflow", SciPlex3SamplingV5OverflowError, "panel total"),
        ("feature-allocation", SciPlex3SamplingV5OverflowError, "feature allocation"),
        ("zero-panel", SciPlex3SamplingV5Error, "zero panel"),
    ),
)
def test_rng_failures_and_impossible_zero_panel_fail_closed(
    failure: str,
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        sampling_module._sample_positive_row(
            np.asarray((1.0,)),
            1.0,
            np.ones((1, 1)),
            _AdversarialGenerator(failure),  # type: ignore[arg-type]
        )
