from __future__ import annotations

import ast
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

WORK_ROOT = Path(__file__).resolve().parent
DRIVER_PATH = WORK_ROOT / "item12_1_local_map_plate_context_diagnostic.py"
PARENT_DRIVER_PATH = WORK_ROOT / "item12_v4_nonissuing_trajectory.py"


def _load_driver() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "_item12_1_local_map_plate_context_diagnostic_tested",
        DRIVER_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


driver = _load_driver()


def _gamma_entropy(shape: np.ndarray, rate: np.ndarray) -> np.ndarray:
    from scipy.special import digamma, gammaln

    return np.asarray(
        shape - np.log(rate) + gammaln(shape) + (1.0 - shape) * digamma(shape),
        dtype=np.float64,
    )


def _synthetic_candidate() -> SimpleNamespace:
    from scipy.special import digamma, gammaln

    class CandidateError(ValueError):
        pass

    class LocalState:
        def __init__(self, theta_shape: np.ndarray, theta_rate: np.ndarray) -> None:
            self.theta_shape = theta_shape
            self.theta_rate = theta_rate

    class Validated:
        def __init__(self, wells: tuple[object, ...]) -> None:
            self.wells = wells

    return SimpleNamespace(
        np=np,
        digamma=digamma,
        gammaln=gammaln,
        _gamma_entropy=_gamma_entropy,
        SciPlex3CandidateError=CandidateError,
        _LocalVariationalState=LocalState,
        _ValidatedTrainingDesign=Validated,
        SCIPLEX3_CANDIDATE_BATCH_SIZE=512,
        SCIPLEX3_CANDIDATE_ELBO_DECREASE_RTOL=1e-8,
        SCIPLEX3_CANDIDATE_FACTOR_COUNT=16,
        SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE=0.1,
        SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL=1e-8,
        SCIPLEX3_CANDIDATE_MASS_EPS_MULTIPLIER=64.0,
        SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS=50,
        SCIPLEX3_CANDIDATE_PLATE_COUNT=8,
        SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT=2,
        SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT=1,
        SCIPLEX3_FEATURE_COUNT=5,
    )


def _make_synthetic_failure(
    defect: str | None = None,
) -> tuple[object, object, BaseException, BaseException, dict[str, Any]]:
    candidate = _synthetic_candidate()

    class RunnerError(RuntimeError):
        pass

    runner = SimpleNamespace(SciPlex3CandidateRunnerError=RunnerError)
    holder: dict[str, Any] = {}
    factor_count = candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT
    feature_count = candidate.SCIPLEX3_FEATURE_COUNT
    loading_concentration = np.fromfunction(
        lambda factor, feature: 0.3 + 0.01 * (factor + 1.0) * (feature + 2.0),
        (factor_count, feature_count),
        dtype=float,
    )
    prior_mean = np.linspace(0.75, 2.25, factor_count, dtype=np.float64)
    well_factor_means = prior_mean[None, :]
    rho_raw = np.fromfunction(
        lambda plate, factor: np.exp(0.01 * (plate - 3.5) * (factor + 1.0)),
        (8, factor_count),
        dtype=float,
    )
    rho = rho_raw / np.mean(rho_raw, axis=0, keepdims=True)
    counts = SimpleNamespace(row_count=2)
    validated = candidate._ValidatedTrainingDesign((SimpleNamespace(counts=counts),))

    def cavi_pass(
        validated: object,
        state: object,
        loading_concentration: np.ndarray,
        well_factor_means: np.ndarray,
    ) -> None:
        expected_log_loading = (
            candidate.digamma(loading_concentration)
            - candidate.digamma(np.sum(loading_concentration, axis=1))[:, None]
        )
        inner_sweep_count_histogram = [0] * candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS
        maximum_inner_sweeps = 0
        well_index = 0
        row_count = 2
        offset = 0
        batch_start = 0
        batch_stop = 2
        global_slice = slice(0, 2)
        row_for_entry = np.asarray([0, 0, 1, 1], dtype=np.int64)
        feature_indices = np.asarray([0, 3, 1, 4], dtype=np.int64)
        values = np.asarray([7.0, 2.0, 5.0, 4.0], dtype=np.float64)
        prior_mean = well_factor_means[well_index]
        fixed_rate = candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE / prior_mean + 1.0
        batch_rate = np.broadcast_to(fixed_rate, (2, factor_count))
        state.theta_rate[global_slice] = batch_rate
        passing_streak = 0
        terminal_shape_residual = math.inf
        terminal_elog_residual = math.inf
        for sweep_count in range(1, candidate.SCIPLEX3_CANDIDATE_MAX_INNER_SWEEPS + 1):
            if sweep_count < 1:
                raise AssertionError("unreachable invalid synthetic sweep")
            current_shape = np.asarray(state.theta_shape[global_slice], dtype=np.float64, order="C")
            expected_log_theta = candidate.digamma(current_shape) - np.log(batch_rate)
            logits = expected_log_theta[row_for_entry] + expected_log_loading[:, feature_indices].T
            logits -= np.max(logits, axis=1, keepdims=True)
            responsibilities = np.exp(logits)
            responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
            allocations = values[:, None] * responsibilities
            allocated_counts = np.zeros((2, factor_count), dtype=np.float64)
            np.add.at(allocated_counts, row_for_entry, allocations)
            proposal_shape = candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE + allocated_counts
            denominator = np.maximum(1.0, np.maximum(np.abs(current_shape), np.abs(proposal_shape)))
            terminal_shape_residual = float(
                np.max(np.abs(proposal_shape - current_shape) / denominator)
            )
            terminal_elog_residual = float(
                np.max(np.abs(candidate.digamma(proposal_shape) - candidate.digamma(current_shape)))
            )
            state.theta_shape[global_slice] = proposal_shape
        if (
            sweep_count != 50
            or passing_streak != 0
            or maximum_inner_sweeps != 0
            or sum(inner_sweep_count_histogram) != 0
            or not math.isfinite(terminal_shape_residual)
            or not math.isfinite(terminal_elog_residual)
            or (row_count, offset, batch_start, batch_stop) != (2, 0, 0, 2)
        ):
            raise AssertionError("synthetic traceback witness construction drifted")
        if defect == "validated_identity":
            validated = object()
        elif defect == "state_identity":
            state = object()
        elif defect == "loading_identity":
            loading_concentration = loading_concentration.copy()
        elif defect == "well_means_identity":
            well_factor_means = well_factor_means.copy()
        elif defect == "prior_mean":
            prior_mean = prior_mean.copy()
            prior_mean[0] += 1.0
        elif defect == "offset":
            offset = 1
        elif defect == "batch_start":
            batch_start = 1
        elif defect == "batch_stop":
            batch_stop = 1
        elif defect == "global_slice":
            global_slice = slice(1, 3)
        elif defect == "minimum_sweep":
            inner_sweep_count_histogram[0] = 1
        elif defect == "histogram_sum":
            inner_sweep_count_histogram[1] = 1
            maximum_inner_sweeps = 2
        elif defect == "maximum_sweep":
            maximum_inner_sweeps = 2
        if any(
            value is None
            for value in (
                validated,
                state,
                loading_concentration,
                well_factor_means,
                prior_mean,
            )
        ):
            raise AssertionError("synthetic frame witness unexpectedly became null")
        raise candidate.SciPlex3CandidateError(
            "candidate inner CAVI failed to converge within 50 sweeps"
        )

    candidate._cavi_pass = cavi_pass

    def exact_fit() -> None:
        trace: list[object] = []
        state = candidate._LocalVariationalState(
            np.full((2, factor_count), 0.1, dtype=np.float64),
            np.full((2, factor_count), 2.0, dtype=np.float64),
        )
        holder["state"] = state
        assert not trace and validated is not None and rho is not None
        cavi_pass(validated, state, loading_concentration, well_factor_means)

    candidate._fit_sciplex3_candidate_exact = exact_fit
    cause: BaseException | None = None
    outer: BaseException | None = None
    try:
        exact_fit()
    except candidate.SciPlex3CandidateError as caught:
        cause = caught
        try:
            raise RunnerError("exact candidate fitting failed closed") from caught
        except RunnerError as wrapped:
            outer = wrapped
    assert cause is not None and outer is not None
    return candidate, runner, outer, cause, holder


def _failure_frame_locals(
    candidate: object, cause: BaseException
) -> tuple[dict[str, object], dict[str, object]]:
    fit_locals: dict[str, object] | None = None
    cavi_locals: dict[str, object] | None = None
    cursor = cause.__traceback__
    while cursor is not None:
        if cursor.tb_frame.f_code is candidate._fit_sciplex3_candidate_exact.__code__:
            fit_locals = cursor.tb_frame.f_locals
        if cursor.tb_frame.f_code is candidate._cavi_pass.__code__:
            cavi_locals = cursor.tb_frame.f_locals
        cursor = cursor.tb_next
    assert fit_locals is not None and cavi_locals is not None
    return fit_locals, cavi_locals


def _bind_synthetic_oracle(
    monkeypatch: pytest.MonkeyPatch,
    candidate: object,
    cause: BaseException,
) -> tuple[dict[str, object], dict[str, object]]:
    fit_locals, cavi_locals = _failure_frame_locals(candidate, cause)
    hashes = driver._attempt2_full_state_hashes(fit_locals, candidate)
    monkeypatch.setattr(
        driver,
        "EXPECTED_ATTEMPT2_TERMINAL_SHAPE_RESIDUAL_HEX",
        cavi_locals["terminal_shape_residual"].hex(),
    )
    monkeypatch.setattr(
        driver,
        "EXPECTED_ATTEMPT2_TERMINAL_ELOG_RESIDUAL_HEX",
        cavi_locals["terminal_elog_residual"].hex(),
    )
    monkeypatch.setattr(
        driver, "EXPECTED_ATTEMPT2_THETA_SHAPE_SHA256", hashes["theta_shape_sha256"]
    )
    monkeypatch.setattr(driver, "EXPECTED_ATTEMPT2_THETA_RATE_SHA256", hashes["theta_rate_sha256"])
    monkeypatch.setattr(
        driver,
        "EXPECTED_ATTEMPT2_LOADING_CONCENTRATION_SHA256",
        hashes["loading_concentration_sha256"],
    )
    monkeypatch.setattr(
        driver,
        "EXPECTED_ATTEMPT2_WELL_FACTOR_MEANS_SHA256",
        hashes["well_factor_means_sha256"],
    )
    monkeypatch.setattr(driver, "EXPECTED_ATTEMPT2_RHO_SHA256", hashes["rho_sha256"])
    return fit_locals, cavi_locals


def _assert_no_arrays(value: object) -> None:
    assert not isinstance(value, np.ndarray)
    if isinstance(value, dict):
        for key, item in value.items():
            assert type(key) is str
            _assert_no_arrays(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_arrays(item)


def _manual_local_map_fixture(candidate: object) -> tuple[object, np.ndarray]:
    factor_count = candidate.SCIPLEX3_CANDIDATE_FACTOR_COUNT
    prior_mean = np.linspace(0.8, 2.3, factor_count, dtype=np.float64)
    batch_rate = (candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE / prior_mean + 1.0)[None, :]
    expected_log_loading = np.fromfunction(
        lambda factor, feature: -0.09 * (factor + 1.0) + 0.04 * (feature + 1.0),
        (factor_count, candidate.SCIPLEX3_FEATURE_COUNT),
        dtype=float,
    )
    replay_input = SimpleNamespace(
        batch_start=0,
        batch_rate=batch_rate,
        expected_log_loading=expected_log_loading,
        feature_indices=np.asarray([0, 3], dtype=np.int64),
        prior_mean=prior_mean,
        row_for_entry=np.asarray([0, 0], dtype=np.int64),
        values=np.asarray([3.0, 5.0], dtype=np.float64),
        well_row_count=1,
    )
    current = np.linspace(0.25, 1.75, factor_count, dtype=np.float64)[None, :]
    return replay_input, current


def test_lineage_constants_and_import_are_source_free_and_lazy() -> None:
    assert driver._file_sha256(PARENT_DRIVER_PATH) == driver.ATTEMPT2_DRIVER_SHA256_ORACLE
    assert driver.ATTEMPT2_REPORT_SHA256_ORACLE == (
        "66e9debc1a402e7aa68cbc934f7c5f641529eea3187ec15606364c912af8faa8"
    )
    source = DRIVER_PATH.read_text()
    assert "/Volumes/Databank" not in source
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util,sys;"
                f"s=importlib.util.spec_from_file_location('d',{str(DRIVER_PATH)!r});"
                "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
                "print('scipy.special' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout == "False\n"


def test_static_timer_and_poisons_span_capture_clear_and_replay() -> None:
    source = DRIVER_PATH.read_text()
    tree = ast.parse(source)
    run = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run"
    )
    run_source = ast.get_source_segment(source, run)
    assert run_source is not None
    installed = run_source.index("_install_nonissuance_poisons")
    armed = run_source.index("signal.setitimer(signal.ITIMER_REAL, float(FIT_WALL_LIMIT_SECONDS))")
    consumed = run_source.index("_consume_item12_1_initial_failure")
    disarmed = run_source.index("signal.setitimer(signal.ITIMER_REAL, 0.0)")
    restored = run_source.index("poisons.restore()", consumed + 1)
    measured = run_source.index("fit_elapsed_seconds = time.monotonic() - fit_started")
    assert installed < armed < consumed < disarmed < restored < measured


def test_synthetic_failure_replay_clears_tracebacks_isolates_and_wipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, runner, outer, cause, holder = _make_synthetic_failure()
    _bind_synthetic_oracle(monkeypatch, candidate, cause)
    state_hash_before = driver._canonical_array_sha256(holder["state"].theta_shape, np)
    original_evaluate = driver._evaluate_local_map
    map_calls: list[tuple[str, str]] = []

    def counted_evaluate(
        replay_input: object, current_shape: object, active_candidate: object
    ) -> object:
        if len(map_calls) >= 51:
            raise AssertionError("the replay attempted to evaluate A52")
        input_digest = driver._canonical_array_sha256(current_shape, np)
        result = original_evaluate(replay_input, current_shape, active_candidate)
        map_calls.append(
            (
                input_digest,
                driver._canonical_array_sha256(result.proposal_shape, np),
            )
        )
        return result

    monkeypatch.setattr(driver, "_evaluate_local_map", counted_evaluate)
    failure, diagnostic = driver._consume_item12_1_initial_failure(outer, candidate, runner)
    assert failure["stage"] == "initial-untraced-equilibration"
    assert diagnostic["pass"] is True
    assert diagnostic["tracebacks_and_causes_cleared_before_replay"] is True
    assert diagnostic["raw_replay_copies_destroyed_before_report"] is True
    assert diagnostic["traceback_copy_manifest"]["copies_isolated"] is True
    assert diagnostic["production_sweep_50_replay"]["all_exact"] is True
    assert diagnostic["terminal_state_exact_traceback_current_proposal_and_state_slice"] is True
    assert diagnostic["per_sweep"][0]["invariants"]["incoming_shape_mass_pass"] is True
    assert diagnostic["local_map_evaluation_count"] == 51 == len(map_calls)
    assert diagnostic["map_state_digest_chain_exact"] is True
    assert len(diagnostic["per_sweep"]) == 50
    assert len(diagnostic["per_sweep_max_row_jacobian_spectral_radii"]) == 50
    assert [item["sweep"] for item in diagnostic["per_sweep"]] == list(range(1, 51))
    assert [item["input_state_index"] for item in diagnostic["per_sweep"]] == list(range(50))
    assert [item["output_state_index"] for item in diagnostic["per_sweep"]] == list(range(1, 51))
    assert [
        item["sweep"] for item in diagnostic["per_sweep_max_row_jacobian_spectral_radii"]
    ] == list(range(1, 51))
    assert [
        item["input_state_index"]
        for item in diagnostic["per_sweep_max_row_jacobian_spectral_radii"]
    ] == list(range(50))
    assert diagnostic["terminal_state_jacobian_spectral_radius"]["state_index"] == 50
    assert diagnostic["terminal_state_jacobian_spectral_radius"]["production_sweep"] is None
    assert [call[1] for call in map_calls[:-1]] == [call[0] for call in map_calls[1:]]
    assert [item["shape_sha256"] for item in diagnostic["per_sweep"]] == [
        call[0] for call in map_calls[:50]
    ]
    assert [item["outgoing_proposal_shape_sha256"] for item in diagnostic["per_sweep"]] == [
        call[1] for call in map_calls[:50]
    ]
    assert diagnostic["diagnostic_lookahead_F_A50_to_A51"]["input_shape_sha256"] == map_calls[50][0]
    assert (
        diagnostic["diagnostic_lookahead_F_A50_to_A51"]["proposal_A51_shape_sha256"]
        == map_calls[50][1]
    )
    assert all(item["gates_pass"] for item in diagnostic["plate_contexts"])
    assert len({item["normalized_basis_row_sha256"] for item in diagnostic["plate_contexts"]}) == 16
    assert outer.__traceback__ is None and outer.__cause__ is None
    assert cause.__traceback__ is None
    assert driver._canonical_array_sha256(holder["state"].theta_shape, np) == state_hash_before
    _assert_no_arrays(diagnostic)
    driver._canonical_json_bytes(diagnostic)


def test_exact_baseline_mismatch_fails_before_replay_and_still_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, runner, outer, cause, _holder = _make_synthetic_failure()
    _bind_synthetic_oracle(monkeypatch, candidate, cause)
    monkeypatch.setattr(driver, "EXPECTED_ATTEMPT2_THETA_SHAPE_SHA256", "0" * 64)
    replay_called = False

    def forbidden_replay(*_args: object, **_kwargs: object) -> object:
        nonlocal replay_called
        replay_called = True
        raise AssertionError("replay must not run after baseline mismatch")

    monkeypatch.setattr(driver, "_replay_local_map_diagnostic", forbidden_replay)
    with pytest.raises(driver.TrajectoryError, match="full state differs"):
        driver._consume_item12_1_initial_failure(outer, candidate, runner)
    assert replay_called is False
    assert outer.__traceback__ is None and outer.__cause__ is None
    assert cause.__traceback__ is None


def test_traceback_clear_proof_failure_blocks_replay_and_wipes_private_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, runner, outer, cause, _holder = _make_synthetic_failure()
    _bind_synthetic_oracle(monkeypatch, candidate, cause)
    replay_called = False
    wiped = False
    original_wipe = driver._wipe_replay_input

    def forbidden_replay(*_args: object, **_kwargs: object) -> object:
        nonlocal replay_called
        replay_called = True
        raise AssertionError("replay must not run without an exact traceback-clear proof")

    def recording_wipe(replay_input: object, np_module: object) -> bool:
        nonlocal wiped
        result = original_wipe(replay_input, np_module)
        wiped = result and all(
            np.asarray(getattr(replay_input, name)).size == 0
            for name in (
                "expected_log_loading",
                "batch_rate",
                "row_for_entry",
                "feature_indices",
                "values",
                "prior_mean",
                "rho",
                "traceback_current_shape",
                "traceback_proposal_shape",
                "traceback_allocated_counts",
                "traceback_responsibilities",
                "traceback_allocations",
                "traceback_state_terminal_shape",
            )
        )
        return result

    monkeypatch.setattr(driver.traceback_module, "clear_frames", lambda _traceback: None)
    monkeypatch.setattr(driver, "_replay_local_map_diagnostic", forbidden_replay)
    monkeypatch.setattr(driver, "_wipe_replay_input", recording_wipe)
    with pytest.raises(driver.TrajectoryError, match="cleanup proof failed before local-map"):
        driver._consume_item12_1_initial_failure(outer, candidate, runner)
    assert replay_called is False
    assert wiped is True


@pytest.mark.parametrize(
    "defect",
    [
        "validated_identity",
        "state_identity",
        "loading_identity",
        "well_means_identity",
        "prior_mean",
        "offset",
        "batch_start",
        "batch_stop",
        "global_slice",
        "minimum_sweep",
        "histogram_sum",
        "maximum_sweep",
    ],
)
def test_frame_identity_order_and_prior_schedule_mismatches_fail_before_replay_and_clear(
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    candidate, runner, outer, cause, _holder = _make_synthetic_failure(defect)
    _bind_synthetic_oracle(monkeypatch, candidate, cause)
    replay_called = False

    def forbidden_replay(*_args: object, **_kwargs: object) -> object:
        nonlocal replay_called
        replay_called = True
        raise AssertionError("replay must not run after a frame/schedule mismatch")

    monkeypatch.setattr(driver, "_replay_local_map_diagnostic", forbidden_replay)
    with pytest.raises(driver.TrajectoryError):
        driver._consume_item12_1_initial_failure(outer, candidate, runner)
    assert replay_called is False
    assert outer.__traceback__ is None and outer.__cause__ is None and outer.__context__ is None
    assert cause.__traceback__ is None and cause.__cause__ is None and cause.__context__ is None


def test_objective_cosine_severity_jacobian_and_rho_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, runner, outer, cause, _holder = _make_synthetic_failure()
    _bind_synthetic_oracle(monkeypatch, candidate, cause)
    _failure, diagnostic = driver._consume_item12_1_initial_failure(outer, candidate, runner)
    first = diagnostic["per_sweep"][0]
    assert first["synchronized_local_variational_objective"] == first["post_coordinate_objective"]
    assert first["shape_coordinate_objective_change"]["production_material_decrease"] is False
    assert first["worst_row_factor"]["severity_over_tolerance"]["value"] == max(
        first["worst_row_factor"]["Rshape"]["value"] / 1e-8,
        first["worst_row_factor"]["Relog"]["value"] / 1e-8,
    )
    assert "current_shape" in first["worst_row_factor"]
    assert "proposal_shape" in first["worst_row_factor"]
    assert "allocated_share_of_row_umi" in first["worst_row_factor"]
    terminal = diagnostic["terminal_state_jacobian_spectral_radius"]
    assert terminal["radius"]["methods_agree_within_ambiguity"] is True
    assert len(terminal["radius"]["symmetric_eigenvalue_sha256"]) == 64
    assert len(terminal["radius"]["generic_eigenvalue_sha256"]) == 64
    assert len(diagnostic["two_step_row_jacobian_spectral_radius"]["eigenvalue_sha256"]) == 64
    assert all(
        item["factorwise_mean_one_abs_error"]["value"] <= 5e-13
        and item["share_sum_abs_error"]["value"] <= 8 * np.finfo(np.float64).eps
        and 1.0 <= item["effective_context_count_inverse_simpson"]["value"] <= 8.0
        for item in diagnostic["plate_contexts"]
    )


def test_local_map_and_weighted_objective_match_independent_scalar_formula() -> None:
    candidate = _synthetic_candidate()
    replay_input, current = _manual_local_map_fixture(candidate)
    evaluation = driver._evaluate_local_map(replay_input, current, candidate)
    expected_log_theta = candidate.digamma(current) - np.log(replay_input.batch_rate)
    manual_responsibilities = np.empty((2, 16), dtype=np.float64)
    for entry_index, feature_index in enumerate(replay_input.feature_indices):
        logits = np.asarray(
            [
                expected_log_theta[0, factor_index]
                + replay_input.expected_log_loading[factor_index, feature_index]
                for factor_index in range(16)
            ],
            dtype=np.float64,
        )
        shifted = logits - max(float(value) for value in logits)
        exponentials = np.asarray([math.exp(float(value)) for value in shifted])
        manual_responsibilities[entry_index] = exponentials / math.fsum(
            float(value) for value in exponentials
        )
    manual_allocations = replay_input.values[:, None] * manual_responsibilities
    manual_counts = np.sum(manual_allocations, axis=0, keepdims=True)
    manual_proposal = candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE + manual_counts
    np.testing.assert_allclose(
        evaluation.responsibilities, manual_responsibilities, rtol=0, atol=2e-16
    )
    np.testing.assert_allclose(evaluation.allocations, manual_allocations, rtol=0, atol=2e-15)
    np.testing.assert_allclose(evaluation.allocated_counts, manual_counts, rtol=0, atol=2e-15)
    np.testing.assert_allclose(evaluation.proposal_shape, manual_proposal, rtol=0, atol=2e-15)

    def scalar_objective(shape: np.ndarray) -> float:
        rate = replay_input.batch_rate
        elog = candidate.digamma(shape) - np.log(rate)
        theta_mean = shape / rate
        terms: list[float] = []
        for factor_index in range(16):
            terms.append(float(manual_counts[0, factor_index] * elog[0, factor_index]))
            terms.append(-float(theta_mean[0, factor_index]))
        for entry_index, feature_index in enumerate(replay_input.feature_indices):
            terms.append(-math.lgamma(float(replay_input.values[entry_index]) + 1.0))
            for factor_index in range(16):
                allocated = float(manual_allocations[entry_index, factor_index])
                probability = float(manual_responsibilities[entry_index, factor_index])
                terms.append(
                    allocated
                    * float(replay_input.expected_log_loading[factor_index, feature_index])
                )
                terms.append(-allocated * math.log(probability))
        fixed_shape = candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE
        for factor_index in range(16):
            shape_value = float(shape[0, factor_index])
            rate_value = float(rate[0, factor_index])
            prior_value = float(replay_input.prior_mean[factor_index])
            elog_value = float(elog[0, factor_index])
            theta_value = float(theta_mean[0, factor_index])
            terms.append(
                fixed_shape * (math.log(fixed_shape) - math.log(prior_value))
                - math.lgamma(fixed_shape)
                + (fixed_shape - 1.0) * elog_value
                - fixed_shape * theta_value / prior_value
            )
            terms.append(
                shape_value
                - math.log(rate_value)
                + math.lgamma(shape_value)
                + (1.0 - shape_value) * float(candidate.digamma(shape_value))
            )
        omega = candidate.SCIPLEX3_CANDIDATE_TRAINING_RECORD_COUNT / (
            candidate.SCIPLEX3_CANDIDATE_TRAINING_WELL_COUNT * replay_input.well_row_count
        )
        return omega * math.fsum(terms)

    manual_pre = scalar_objective(current)
    manual_post = scalar_objective(manual_proposal)
    assert evaluation.pre_shape_objective_fsum == pytest.approx(manual_pre, abs=2e-13)
    assert evaluation.post_coordinate_objective_fsum == pytest.approx(manual_post, abs=2e-13)
    assert evaluation.pre_shape_objective == pytest.approx(manual_pre, abs=2e-13)
    assert evaluation.post_coordinate_objective == pytest.approx(manual_post, abs=2e-13)
    assert manual_post > manual_pre


def test_objective_thresholds_and_cosine_use_exact_ordered_contract() -> None:
    candidate = _synthetic_candidate()
    previous = 100.0
    threshold = candidate.SCIPLEX3_CANDIDATE_ELBO_DECREASE_RTOL * previous
    band = 64.0 * np.finfo(np.float64).eps * previous
    clear = driver._objective_change_diagnostic(
        previous,
        previous - threshold - 4.0 * band,
        previous,
        previous - threshold - 4.0 * band,
        candidate,
    )
    assert clear["production_material_decrease"] is True
    assert clear["fsum_material_decrease"] is True
    assert clear["boundary_or_summation_ambiguous"] is False
    assert clear["production_ambiguity_band"]["value"] == band
    assert clear["fsum_ambiguity_band"]["value"] == band
    boundary = driver._objective_change_diagnostic(
        previous, previous - threshold, previous, previous - threshold, candidate
    )
    assert boundary["boundary_or_summation_ambiguous"] is True
    disagreement = driver._objective_change_diagnostic(
        previous,
        previous - threshold - 4.0 * band,
        previous,
        previous,
        candidate,
    )
    assert disagreement["production_material_decrease"] is True
    assert disagreement["fsum_material_decrease"] is False
    assert disagreement["boundary_or_summation_ambiguous"] is True

    current = np.asarray([[1.0e16, 1.0, -1.0e16], [4.0, -3.0, 2.0]])
    one_step = current + np.asarray([[2.0, -0.5, -2.0], [1.0, 2.0, -1.0]])
    two_step = one_step + np.asarray([[-1.0, 0.25, 1.0], [-0.5, -3.0, 2.0]])
    cosine, sign = driver._update_cosine_sign(current, one_step, two_step, np)
    first = (one_step - current) / np.maximum(1.0, np.maximum(np.abs(current), np.abs(one_step)))
    second = (two_step - one_step) / np.maximum(1.0, np.maximum(np.abs(one_step), np.abs(two_step)))
    first_flat = first.ravel(order="C")
    second_flat = second.ravel(order="C")
    dot = math.fsum(
        float(first_flat[index]) * float(second_flat[index]) for index in range(first_flat.size)
    )
    expected_cosine = dot / (
        math.sqrt(math.fsum(float(value) ** 2 for value in first_flat))
        * math.sqrt(math.fsum(float(value) ** 2 for value in second_flat))
    )
    assert cosine == expected_cosine
    assert sign == (-1 if dot < 0.0 else (1 if dot > 0.0 else 0))


def test_scientific_route_requires_objective_integrity_before_solver() -> None:
    common = {
        "core_replay_valid": True,
        "strict_two_cycle": True,
        "jacobian_noncontractive": True,
        "jacobian_ambiguous": False,
        "context_collapsed_or_tiny": False,
        "context_ambiguous": False,
    }
    decrease = driver._scientific_route_branches(
        **common,
        objective_material_decrease=True,
        objective_ambiguous=False,
    )
    assert "implementation-fix" in decrease
    assert "v5-safeguarded-local-solver" not in decrease
    ambiguity = driver._scientific_route_branches(
        **common,
        objective_material_decrease=False,
        objective_ambiguous=True,
    )
    assert "v5-safeguarded-local-solver" not in ambiguity
    assert "ambiguity-no-conclusion-for-affected-gate" in ambiguity
    intact = driver._scientific_route_branches(
        **common,
        objective_material_decrease=False,
        objective_ambiguous=False,
    )
    assert "v5-safeguarded-local-solver" in intact


def test_worst_row_factor_uses_severity_and_c_order_tie() -> None:
    candidate = _synthetic_candidate()
    shape = np.ones((2, 16), dtype=np.float64)
    proposal = shape.copy()
    proposal[0, 0] = 2.0
    proposal[0, 1] = 2.0
    evaluation = SimpleNamespace(
        proposal_shape=proposal,
        allocated_counts=proposal - candidate.SCIPLEX3_CANDIDATE_FIXED_FACTOR_SHAPE,
    )
    replay_input = SimpleNamespace(
        batch_start=7,
        batch_rate=np.full((2, 16), 1.5),
        prior_mean=np.ones(16),
        row_for_entry=np.asarray([0, 1]),
        values=np.asarray([2.0, 3.0]),
    )
    summary = driver._worst_row_factor_summary(replay_input, shape, evaluation, candidate)
    assert summary["batch_row_index"] == 0
    assert summary["factor_index"] == 0
    assert summary["well_row_index"] == 7
    expected_severity = max(
        summary["Rshape"]["value"] / candidate.SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL,
        summary["Relog"]["value"] / candidate.SCIPLEX3_CANDIDATE_INNER_RESIDUAL_TOL,
    )
    assert summary["severity_over_tolerance"]["value"] == expected_severity


def test_analytic_jacobian_matches_direct_construction_and_finite_difference() -> None:
    from scipy.special import polygamma

    candidate = _synthetic_candidate()
    replay_input, current = _manual_local_map_fixture(candidate)
    evaluation = driver._evaluate_local_map(replay_input, current, candidate)
    jacobian = driver._row_jacobians(replay_input, current, evaluation, candidate, polygamma)[0]
    hessian = np.zeros((16, 16), dtype=np.float64)
    for value, probabilities in zip(replay_input.values, evaluation.responsibilities, strict=True):
        hessian += float(value) * (np.diag(probabilities) - np.outer(probabilities, probabilities))
    direct = hessian @ np.diag(polygamma(1, current[0]))
    np.testing.assert_allclose(jacobian, direct, rtol=2e-14, atol=2e-14)
    finite_difference = np.empty_like(jacobian)
    step = 1e-6
    for factor_index in range(16):
        plus = current.copy()
        minus = current.copy()
        plus[0, factor_index] += step
        minus[0, factor_index] -= step
        plus_map = driver._evaluate_local_map(replay_input, plus, candidate).proposal_shape[0]
        minus_map = driver._evaluate_local_map(replay_input, minus, candidate).proposal_shape[0]
        finite_difference[:, factor_index] = (plus_map - minus_map) / (2.0 * step)
    np.testing.assert_allclose(jacobian, finite_difference, rtol=2e-6, atol=2e-7)

    crafted = np.zeros((16, 16), dtype=np.float64)
    crafted[0, 0] = -5.0
    crafted[1, 1] = 2.0
    radius = driver._jacobian_radius(crafted, np.ones(16), candidate, polygamma)
    assert radius["spectral_radius"]["value"] == pytest.approx(2.0)
    assert radius["generic_eigenvalue_radius"]["value"] == pytest.approx(5.0)
    assert radius["methods_agree_within_ambiguity"] is False

    first = np.eye(16)
    second = np.eye(16)
    first[:2, :2] = np.asarray([[1.0, 2.0], [0.0, 1.0]])
    second[:2, :2] = np.asarray([[1.0, 0.0], [3.0, 1.0]])
    composition = driver._two_step_jacobian_summary(
        SimpleNamespace(batch_start=0), [first], [second], candidate
    )
    expected = second @ first
    reverse = first @ second
    assert composition["composition_order"] == "J(A50)@J(A49)"
    assert composition["jacobian_sha256"] == driver._canonical_array_sha256(expected, np)
    assert composition["jacobian_sha256"] != driver._canonical_array_sha256(reverse, np)


def test_rho_inverse_simpson_scalars_ties_boundaries_and_tiny_share() -> None:
    candidate = _synthetic_candidate()
    rho = np.ones((8, 16), dtype=np.float64)
    tied_shares = np.asarray([0.2, 0.2, 0.15, 0.1, 0.1, 0.1, 0.075, 0.075])
    boundary_shares = np.asarray([0.5, *(0.5 / 7.0 for _ in range(7))])
    tiny = math.sqrt(np.finfo(np.float64).eps) / 2.0
    tiny_shares = np.asarray([1.0 - 7.0 * tiny, *(tiny for _ in range(7))])
    mixed_second = (0.8 + math.sqrt(19.68)) / 14.0
    mixed_remainder = (0.4 - mixed_second) / 6.0
    mixed_shares = np.asarray([0.6, mixed_second, *(mixed_remainder for _ in range(6))])
    rho[:, 0] = 8.0 * tied_shares
    rho[:, 1] = 8.0 * boundary_shares
    rho[:, 2] = 8.0 * tiny_shares
    rho[:, 3] = 8.0 * mixed_shares
    replay_input = SimpleNamespace(
        normalized_basis_row_sha256=tuple(f"{index:064x}" for index in range(16)),
        rho=rho,
    )
    diagnostics = driver._rho_effective_context_diagnostics(replay_input, candidate)
    first = diagnostics[0]
    total = math.fsum(float(value) for value in rho[:, 0])
    squared = math.fsum(float(value) * float(value) for value in rho[:, 0])
    shares = rho[:, 0] / total
    assert first["factorwise_sum"]["value"] == total
    assert first["factorwise_mean"]["value"] == total / 8.0
    assert first["rho_squared_sum"]["value"] == squared
    assert first["share_sum"]["value"] == math.fsum(float(value) for value in shares)
    assert first["effective_context_count_inverse_simpson"]["value"] == total * total / squared
    assert first["maximum_share_context_index"] == 0
    assert first["maximum_share"]["value"] == shares[0]
    assert diagnostics[1]["collapse_boundary_ambiguous"] is True
    assert diagnostics[2]["collapsed"] is True
    assert diagnostics[2]["tiny_normalized_share"] is True
    assert diagnostics[3]["effective_count_collapse_boundary_ambiguous"] is True
    assert diagnostics[3]["maximum_share_collapse_boundary_ambiguous"] is False
    assert diagnostics[3]["maximum_share_clear_collapse"] is True
    assert diagnostics[3]["collapsed"] is True
    assert all(item["gates_pass"] for item in diagnostics)


def test_strict_two_cycle_requires_both_adjacent_failures() -> None:
    candidate = _synthetic_candidate()
    replay_input = SimpleNamespace(batch_start=0)
    a49 = np.full((1, 16), 1.0)
    a50 = np.full((1, 16), 2.0)
    exact_cycle = driver._strict_two_cycle_summary(replay_input, a49, a50, a49, candidate)
    assert exact_cycle["same_row_strict_two_cycle"] is True
    second_converged = driver._strict_two_cycle_summary(
        replay_input, a49, a50, a50 + 1e-12, candidate
    )
    assert second_converged["same_row_strict_two_cycle"] is False


def test_cli_argument_failure_is_one_finite_canonical_json_line() -> None:
    completed = subprocess.run(
        [sys.executable, str(DRIVER_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 64
    assert completed.stderr == ""
    assert completed.stdout.count("\n") == 1
    payload = completed.stdout.encode().rstrip(b"\n")
    value = json.loads(payload)
    assert driver._canonical_json_bytes(value) == payload
    assert value["schema"] == "sciplex3-item12.1b-local-map-plate-context-replay-harness"
    assert value["status"] == "diagnostic-argument-failure-no-artifact-issued"
