from __future__ import annotations

import math

import numpy as np
import pytest

import cellstate.evaluation.sciplex3_candidate_v5 as objective_module
from cellstate.evaluation.sciplex3_candidate_v5 import (
    SCIPLEX3_V5_ACTION_LIKELIHOOD_SCALE,
    SCIPLEX3_V5_EQUAL_WELL_SCALE,
    SCIPLEX3_V5_GRADIENT_TOL,
    SciPlex3V5AcceptedSubstep,
    SciPlex3V5ActionContextFit,
    SciPlex3V5ActionParameters,
    SciPlex3V5CompleteBlock,
    SciPlex3V5Design,
    SciPlex3V5ObjectiveError,
    SciPlex3V5ObjectiveGradients,
    action_context_parameter_sha256,
    feasible_coordinates_from_log_rho,
    fit_fixed_q_action_context_m_step,
    fixed_q_dose_block_gradient_hessian,
    fixed_q_factor_arrowhead_gradient_hessian,
    fixed_q_full_elbo_action_context,
    fixed_q_full_elbo_action_context_gradients,
    independent_fixed_q_full_elbo_action_context,
    log_rho_from_feasible_coordinates,
)


def _design() -> SciPlex3V5Design:
    action_wells = np.arange(752, dtype=np.int64).reshape(188, 4)
    vehicle_wells = np.arange(752, 768, dtype=np.int64).reshape(8, 2)
    plates = np.empty(768, dtype=np.int64)
    for compound_index in range(188):
        for dose_index in range(4):
            plates[action_wells[compound_index, dose_index]] = (compound_index + dose_index) % 8
    for plate_index in range(8):
        plates[vehicle_wells[plate_index]] = plate_index
    return SciPlex3V5Design(plates, action_wells, vehicle_wells)


def _parameters() -> SciPlex3V5ActionParameters:
    alpha = np.linspace(-0.2, 0.25, 16, dtype=np.float64)
    coordinates = 0.11 * np.sin(np.arange(7 * 16, dtype=np.float64).reshape(7, 16) + 0.3)
    delta = 0.035 * np.sin(np.arange(188 * 4 * 16, dtype=np.float64).reshape(188, 4, 16) / 19.0)
    return SciPlex3V5ActionParameters(
        alpha=alpha,
        log_rho=log_rho_from_feasible_coordinates(coordinates),
        delta=delta,
    )


def _posterior(design: SciPlex3V5Design) -> np.ndarray:
    well = np.arange(768, dtype=np.float64)[:, None]
    factor = np.arange(16, dtype=np.float64)[None, :]
    plate = design.training_well_plate_indices[:, None]
    return np.exp(
        0.09 * np.sin((well + 1.0) * (factor + 2.0) / 61.0)
        + 0.025 * (plate - 3.5)
        + 0.01 * (factor - 7.5)
    )


def _replace_parameters(
    parameters: SciPlex3V5ActionParameters,
    *,
    alpha: np.ndarray | None = None,
    log_rho: np.ndarray | None = None,
    delta: np.ndarray | None = None,
) -> SciPlex3V5ActionParameters:
    return SciPlex3V5ActionParameters(
        alpha=parameters.alpha if alpha is None else alpha,
        log_rho=parameters.log_rho if log_rho is None else log_rho,
        delta=parameters.delta if delta is None else delta,
    )


def _central_difference(function: object, value: np.ndarray, index: tuple[int, ...]) -> float:
    step = 1e-5
    plus = value.copy()
    minus = value.copy()
    plus[index] += step
    minus[index] -= step
    callable_function = function
    assert callable(callable_function)
    return float((callable_function(plus) - callable_function(minus)) / (2.0 * step))


def test_canonical_objective_matches_independent_full_elbo_and_exact_scale() -> None:
    design = _design()
    posterior = _posterior(design)
    parameters = _parameters()

    canonical = fixed_q_full_elbo_action_context(posterior, parameters, design)
    independent = independent_fixed_q_full_elbo_action_context(posterior, parameters, design)

    assert SCIPLEX3_V5_EQUAL_WELL_SCALE == 94_785 / 768
    assert SCIPLEX3_V5_ACTION_LIKELIHOOD_SCALE == (94_785 / 768) * 0.1
    assert canonical == pytest.approx(independent, rel=2e-15)

    unit_parameters = SciPlex3V5ActionParameters(
        alpha=np.zeros(16, dtype=np.float64),
        log_rho=np.zeros((8, 16), dtype=np.float64),
        delta=np.zeros((188, 4, 16), dtype=np.float64),
    )
    unit_posterior = np.ones((768, 16), dtype=np.float64)
    expected = -SCIPLEX3_V5_ACTION_LIKELIHOOD_SCALE * 768 * 16
    assert fixed_q_full_elbo_action_context(unit_posterior, unit_parameters, design) == expected


def test_alpha_feasible_log_rho_and_treated_delta_gradients_match_finite_differences() -> None:
    design = _design()
    posterior = _posterior(design)
    parameters = _parameters()
    gradients = fixed_q_full_elbo_action_context_gradients(posterior, parameters, design)

    def alpha_objective(alpha: np.ndarray) -> float:
        return fixed_q_full_elbo_action_context(
            posterior,
            _replace_parameters(parameters, alpha=alpha),
            design,
        )

    for alpha_index in (0, 7, 15):
        numerical = _central_difference(alpha_objective, parameters.alpha, (alpha_index,))
        assert gradients.alpha[alpha_index] == pytest.approx(numerical, rel=2e-7, abs=4e-6)

    coordinates = feasible_coordinates_from_log_rho(parameters.log_rho)

    def log_rho_objective(free: np.ndarray) -> float:
        return fixed_q_full_elbo_action_context(
            posterior,
            _replace_parameters(parameters, log_rho=log_rho_from_feasible_coordinates(free)),
            design,
        )

    for rho_index in ((0, 0), (3, 7), (6, 15)):
        numerical = _central_difference(log_rho_objective, coordinates, rho_index)
        assert gradients.feasible_log_rho[rho_index] == pytest.approx(numerical, rel=2e-7, abs=4e-6)

    def delta_objective(delta: np.ndarray) -> float:
        return fixed_q_full_elbo_action_context(
            posterior,
            _replace_parameters(parameters, delta=delta),
            design,
        )

    for delta_index in ((0, 0, 0), (17, 2, 7), (187, 3, 15)):
        numerical = _central_difference(delta_objective, parameters.delta, delta_index)
        assert gradients.delta[delta_index] == pytest.approx(numerical, rel=2e-7, abs=4e-6)


def test_canonical_four_dose_newton_hessian_matches_gradient_finite_differences() -> None:
    posterior = np.asarray((0.72, 1.11, 1.63, 2.04), dtype=np.float64)
    plate_intercepts = np.asarray((-0.18, 0.05, 0.22, -0.07), dtype=np.float64)
    delta = np.asarray((0.12, -0.04, 0.09, 0.19), dtype=np.float64)
    gradient, hessian = fixed_q_dose_block_gradient_hessian(posterior, plate_intercepts, delta)
    step = 1e-6
    numerical_hessian = np.empty((4, 4), dtype=np.float64)
    for column in range(4):
        plus = delta.copy()
        minus = delta.copy()
        plus[column] += step
        minus[column] -= step
        plus_gradient, _ = fixed_q_dose_block_gradient_hessian(posterior, plate_intercepts, plus)
        minus_gradient, _ = fixed_q_dose_block_gradient_hessian(posterior, plate_intercepts, minus)
        numerical_hessian[:, column] = (plus_gradient - minus_gradient) / (2.0 * step)

    assert np.allclose(hessian, numerical_hessian, rtol=2e-9, atol=2e-9)
    assert np.array_equal(hessian, hessian.T)
    assert np.all(np.linalg.eigvalsh(hessian) < 0.0)
    assert np.all(np.isfinite(gradient))


def test_joint_arrowhead_hessian_cross_terms_direction_and_newton_residual() -> None:
    design = _design()
    posterior = np.exp(0.2 * np.sin(np.arange(768, dtype=np.float64) / 23.0))
    beta = np.linspace(-0.2, 0.2, 8, dtype=np.float64)
    delta = 0.03 * np.sin(np.arange(752, dtype=np.float64).reshape(188, 4) / 17.0)

    beta_gradient, delta_gradient, hessian = fixed_q_factor_arrowhead_gradient_hessian(
        posterior, beta, delta, design
    )
    gradient = np.concatenate((beta_gradient, delta_gradient.ravel(order="C")))
    generator = np.random.default_rng(20260811)
    direction = generator.normal(size=760)
    direction /= np.linalg.norm(direction)
    step = 2e-6

    def joint_gradient(coordinates: np.ndarray) -> np.ndarray:
        current_beta_gradient, current_delta_gradient, _ = (
            fixed_q_factor_arrowhead_gradient_hessian(
                posterior,
                coordinates[:8],
                coordinates[8:].reshape(188, 4),
                design,
            )
        )
        return np.concatenate((current_beta_gradient, current_delta_gradient.ravel(order="C")))

    coordinates = np.concatenate((beta, delta.ravel(order="C")))
    numerical_direction = (
        joint_gradient(coordinates + step * direction)
        - joint_gradient(coordinates - step * direction)
    ) / (2.0 * step)
    assert np.allclose(numerical_direction, hessian @ direction, rtol=2e-7, atol=2e-7)
    assert np.array_equal(hessian, hessian.T)

    compound_index = 17
    dose_index = 2
    delta_coordinate = 8 + compound_index * 4 + dose_index
    plate_index = int(
        design.training_well_plate_indices[design.action_well_indices[compound_index, dose_index]]
    )
    assert hessian[plate_index, delta_coordinate] < 0.0
    assert hessian[delta_coordinate, plate_index] == hessian[plate_index, delta_coordinate]
    other_compound_coordinate = 8 + (compound_index + 1) * 4 + dose_index
    assert hessian[delta_coordinate, other_compound_coordinate] == 0.0

    beta_step, delta_step, maximum_gradient = objective_module._factor_newton_step(
        posterior, beta, delta, design
    )
    newton_increment = -np.concatenate((beta_step, delta_step.ravel(order="C")))
    residual = hessian @ newton_increment + gradient
    assert np.max(np.abs(residual)) < 1e-9
    assert maximum_gradient == pytest.approx(np.max(np.abs(gradient)), rel=0.0, abs=1e-12)


def test_precision_gate_never_masks_a_one_ulp_elbo_decrease() -> None:
    current = -155_296.88530990639
    one_ulp_worse = float(np.nextafter(current, -math.inf))

    assert objective_module._full_objective_proposal_nondecreases(
        current, current, current, current
    )
    assert objective_module._full_objective_proposal_nondecreases(
        current, float(np.nextafter(current, math.inf)), current, current
    )
    assert not objective_module._full_objective_proposal_nondecreases(
        current, one_ulp_worse, current, current
    )
    assert not objective_module._full_objective_proposal_nondecreases(
        current, current, current, one_ulp_worse
    )
    assert not objective_module._full_objective_proposal_nondecreases(
        current, math.nan, current, current
    )


def test_treated_wells_drive_context_and_complete_m_step_is_independently_monotone() -> None:
    design = _design()
    posterior = np.ones((768, 16), dtype=np.float64)
    plate_effect = np.linspace(-0.3, 0.3, 8, dtype=np.float64)
    dose_effect = np.asarray((-0.15, -0.05, 0.08, 0.22), dtype=np.float64)
    factor_index = np.arange(1, 17, dtype=np.float64)
    for compound_index in range(188):
        for dose_index in range(4):
            well_index = int(design.action_well_indices[compound_index, dose_index])
            plate_index = int(design.training_well_plate_indices[well_index])
            posterior[well_index] = np.exp(
                plate_effect[plate_index]
                + dose_effect[dose_index]
                + 0.03 * np.sin((compound_index + 1) * factor_index)
            )

    fit = fit_fixed_q_action_context_m_step(posterior, design)
    control_only_surrogate = fit_fixed_q_action_context_m_step(np.ones_like(posterior), design)

    assert fit.final_objective > fit.initial_objective
    assert fit.accepted_substeps
    canonical = fit.initial_objective
    independent = fit.initial_independent_objective
    for substep in fit.accepted_substeps:
        assert substep.objective_before == canonical
        assert substep.independent_objective_before == independent
        assert substep.objective_after >= substep.objective_before
        assert substep.independent_objective_after >= substep.independent_objective_before
        assert (
            fixed_q_full_elbo_action_context(posterior, substep.parameters_after, design)
            == substep.objective_after
        )
        assert (
            independent_fixed_q_full_elbo_action_context(
                posterior, substep.parameters_after, design
            )
            == substep.independent_objective_after
        )
        assert substep.parameter_sha256 == action_context_parameter_sha256(substep.parameters_after)
        canonical = substep.objective_after
        independent = substep.independent_objective_after
    assert canonical == fit.final_objective
    assert independent == fit.final_independent_objective
    assert action_context_parameter_sha256(fit.parameters) == (
        fit.accepted_substeps[-1].parameter_sha256
    )
    assert all(block.objective_after >= block.objective_before for block in fit.complete_blocks)
    assert all(
        block.objective_after == pytest.approx(block.independent_objective_after, rel=2e-15)
        for block in fit.complete_blocks
    )
    assert fit.terminal_gradients.maximum_absolute <= SCIPLEX3_V5_GRADIENT_TOL
    assert np.allclose(
        np.mean(np.exp(fit.parameters.log_rho), axis=0),
        1.0,
        rtol=0.0,
        atol=5e-13,
    )
    assert np.max(np.abs(fit.parameters.log_rho)) > 0.1
    assert np.max(np.abs(control_only_surrogate.parameters.log_rho)) < 1e-12
    assert fixed_q_full_elbo_action_context(posterior, fit.parameters, design) == (
        fit.final_objective
    )


@pytest.mark.parametrize("log_standard_deviation", (0.2, 5.0))
def test_full_objective_gate_is_stable_across_log_normal_stiffness(
    log_standard_deviation: float,
) -> None:
    design = _design()
    log_posterior = log_standard_deviation * np.random.default_rng(44).normal(size=(768, 16))
    log_posterior -= np.max(log_posterior, axis=0, keepdims=True)
    posterior = np.exp(log_posterior)
    posterior /= np.mean(posterior, axis=0, keepdims=True)

    fit = fit_fixed_q_action_context_m_step(posterior, design)

    assert fit.final_objective >= fit.initial_objective
    assert fit.final_independent_objective >= fit.initial_independent_objective
    assert fit.terminal_gradients.maximum_absolute <= SCIPLEX3_V5_GRADIENT_TOL
    assert all(
        substep.objective_after >= substep.objective_before
        and substep.independent_objective_after >= substep.independent_objective_before
        for substep in fit.accepted_substeps
    )


def test_topology_gauge_and_posterior_domain_fail_closed() -> None:
    design = _design()
    duplicated = design.action_well_indices.copy()
    duplicated[0, 0] = duplicated[0, 1]
    with pytest.raises(SciPlex3V5ObjectiveError, match="partition"):
        SciPlex3V5Design(
            design.training_well_plate_indices,
            duplicated,
            design.vehicle_well_indices,
        )

    with pytest.raises(SciPlex3V5ObjectiveError, match="mean-one"):
        SciPlex3V5ActionParameters(
            alpha=np.zeros(16, dtype=np.float64),
            log_rho=np.full((8, 16), 0.1, dtype=np.float64),
            delta=np.zeros((188, 4, 16), dtype=np.float64),
        )

    posterior = np.ones((768, 16), dtype=np.float64)
    posterior[0, 0] = 0.0
    with pytest.raises(SciPlex3V5ObjectiveError, match="strictly positive"):
        fixed_q_full_elbo_action_context(posterior, _parameters(), design)

    coordinates = np.zeros((7, 16), dtype=np.float64)
    round_trip = feasible_coordinates_from_log_rho(log_rho_from_feasible_coordinates(coordinates))
    assert np.array_equal(round_trip, coordinates)
    assert math.isfinite(SCIPLEX3_V5_ACTION_LIKELIHOOD_SCALE)


def test_public_objective_types_and_numeric_extremes_fail_closed() -> None:
    design = _design()
    parameters = _parameters()
    posterior = np.ones((768, 16), dtype=np.float64)

    wrong_vehicle_plates = design.training_well_plate_indices.copy()
    wrong_vehicle_plates[int(design.vehicle_well_indices[0, 0])] = 1
    with pytest.raises(SciPlex3V5ObjectiveError, match="declared plate"):
        SciPlex3V5Design(
            wrong_vehicle_plates,
            design.action_well_indices,
            design.vehicle_well_indices,
        )
    with pytest.raises(SciPlex3V5ObjectiveError, match="exact action parameters"):
        action_context_parameter_sha256(object())  # type: ignore[arg-type]
    with pytest.raises(SciPlex3V5ObjectiveError, match="exact action parameters"):
        fixed_q_full_elbo_action_context(posterior, object(), design)  # type: ignore[arg-type]
    with pytest.raises(SciPlex3V5ObjectiveError, match="exact design type"):
        fixed_q_full_elbo_action_context(posterior, parameters, object())  # type: ignore[arg-type]
    with pytest.raises(SciPlex3V5ObjectiveError, match="exact parameters"):
        independent_fixed_q_full_elbo_action_context(
            posterior,
            object(),
            design,  # type: ignore[arg-type]
        )
    with pytest.raises(SciPlex3V5ObjectiveError, match="exact design"):
        independent_fixed_q_full_elbo_action_context(
            posterior,
            parameters,
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(SciPlex3V5ObjectiveError, match="gradients require"):
        fixed_q_full_elbo_action_context_gradients(
            posterior,
            object(),
            design,  # type: ignore[arg-type]
        )
    with pytest.raises(SciPlex3V5ObjectiveError, match="exact design type"):
        fit_fixed_q_action_context_m_step(posterior, object())  # type: ignore[arg-type]
    with pytest.raises(SciPlex3V5ObjectiveError, match="wrong exact type"):
        fit_fixed_q_action_context_m_step(
            posterior,
            design,
            initial=object(),  # type: ignore[arg-type]
        )

    with pytest.raises(SciPlex3V5ObjectiveError, match="posterior means must be positive"):
        fixed_q_dose_block_gradient_hessian(
            np.zeros(4),
            np.zeros(4),
            np.zeros(4),
        )
    with pytest.raises(SciPlex3V5ObjectiveError, match="derivatives are nonfinite"):
        fixed_q_dose_block_gradient_hessian(
            np.ones(4),
            np.full(4, -1_000.0),
            np.zeros(4),
        )
    with pytest.raises(SciPlex3V5ObjectiveError, match="exact design type"):
        fixed_q_factor_arrowhead_gradient_hessian(
            np.ones(768),
            np.zeros(8),
            np.zeros((188, 4)),
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(SciPlex3V5ObjectiveError, match="posterior well means must be positive"):
        fixed_q_factor_arrowhead_gradient_hessian(
            np.zeros(768),
            np.zeros(8),
            np.zeros((188, 4)),
            design,
        )
    with pytest.raises(SciPlex3V5ObjectiveError, match="derivatives are nonfinite"):
        fixed_q_factor_arrowhead_gradient_hessian(
            np.ones(768),
            np.full(8, -1_000.0),
            np.zeros((188, 4)),
            design,
        )

    zero_log_rho = np.zeros((8, 16), dtype=np.float64)
    zero_delta = np.zeros((188, 4, 16), dtype=np.float64)
    nonfinite_predictor = SciPlex3V5ActionParameters(
        alpha=np.full(16, np.finfo(np.float64).max),
        log_rho=zero_log_rho,
        delta=np.full((188, 4, 16), np.finfo(np.float64).max),
    )
    with (
        np.errstate(over="ignore"),
        pytest.raises(SciPlex3V5ObjectiveError, match="linear predictor"),
    ):
        fixed_q_full_elbo_action_context(posterior, nonfinite_predictor, design)

    nonfinite_penalty = SciPlex3V5ActionParameters(
        alpha=np.zeros(16),
        log_rho=zero_log_rho,
        delta=np.full((188, 4, 16), 1e200),
    )
    with np.errstate(over="ignore"), pytest.raises(SciPlex3V5ObjectiveError, match="dose penalty"):
        fixed_q_full_elbo_action_context(posterior, nonfinite_penalty, design)

    nonfinite_objective = SciPlex3V5ActionParameters(
        alpha=np.full(16, -1_000.0),
        log_rho=zero_log_rho,
        delta=zero_delta,
    )
    with pytest.raises(SciPlex3V5ObjectiveError, match="full-ELBO"):
        fixed_q_full_elbo_action_context(posterior, nonfinite_objective, design)
    with pytest.raises(SciPlex3V5ObjectiveError, match="overflowed"):
        independent_fixed_q_full_elbo_action_context(posterior, nonfinite_objective, design)
    with pytest.raises(SciPlex3V5ObjectiveError, match="gradient is nonfinite"):
        fixed_q_full_elbo_action_context_gradients(posterior, nonfinite_objective, design)
    with pytest.raises(SciPlex3V5ObjectiveError, match="Newton derivatives are nonfinite"):
        objective_module._factor_newton_step(
            np.ones(768),
            np.full(8, -1_000.0),
            np.zeros((188, 4)),
            design,
        )


def _zero_gradients() -> SciPlex3V5ObjectiveGradients:
    return SciPlex3V5ObjectiveGradients(
        alpha=np.zeros(16),
        feasible_log_rho=np.zeros((7, 16)),
        delta=np.zeros((188, 4, 16)),
        plate_intercepts=np.zeros((8, 16)),
    )


def _plate_substep(
    *,
    parameters: SciPlex3V5ActionParameters,
    before: float,
    after: float,
) -> SciPlex3V5AcceptedSubstep:
    return SciPlex3V5AcceptedSubstep(
        kind="plate-intercept",
        sweep=1,
        objective_before=before,
        objective_after=after,
        independent_objective_before=before,
        independent_objective_after=after,
        parameters_after=parameters,
        parameter_sha256=action_context_parameter_sha256(parameters),
    )


def _complete_block(*, before: float, after: float) -> SciPlex3V5CompleteBlock:
    return SciPlex3V5CompleteBlock(
        sweep=1,
        objective_before=before,
        objective_after=after,
        independent_objective_before=before,
        independent_objective_after=after,
        maximum_absolute_gradient=0.0,
    )


def test_m_step_witness_tampering_fails_closed_at_each_chain_boundary() -> None:
    parameters = _parameters()
    digest = action_context_parameter_sha256(parameters)
    with pytest.raises(SciPlex3V5ObjectiveError, match="decreased"):
        SciPlex3V5AcceptedSubstep(
            "plate-intercept",
            1,
            -1.0,
            -2.0,
            -1.0,
            -2.0,
            parameters,
            digest,
        )
    with pytest.raises(SciPlex3V5ObjectiveError, match="state witness"):
        SciPlex3V5AcceptedSubstep(
            "plate-intercept",
            1,
            -2.0,
            -1.0,
            -2.0,
            -1.0,
            parameters,
            "0" * 64,
        )
    with pytest.raises(SciPlex3V5ObjectiveError, match="Newton metadata"):
        SciPlex3V5AcceptedSubstep(
            "plate-intercept",
            1,
            -2.0,
            -1.0,
            -2.0,
            -1.0,
            parameters,
            digest,
            factor_index=0,
        )
    with pytest.raises(SciPlex3V5ObjectiveError, match="metadata is invalid"):
        SciPlex3V5AcceptedSubstep(
            "factor-newton",
            1,
            -2.0,
            -1.0,
            -2.0,
            -1.0,
            parameters,
            digest,
            factor_index=16,
            newton_iteration=1,
            step_scale=1.0,
        )
    with pytest.raises(SciPlex3V5ObjectiveError, match="block witness"):
        SciPlex3V5CompleteBlock(1, -2.0, -1.0, -2.0, -1.0, -0.1)
    with pytest.raises(SciPlex3V5ObjectiveError, match="evaluators disagree"):
        SciPlex3V5CompleteBlock(1, -2.0, -1.0, -2.0, -0.5, 0.0)

    substep = _plate_substep(parameters=parameters, before=-2.0, after=-1.0)
    block = _complete_block(before=-2.0, after=-1.0)
    gradients = _zero_gradients()
    with pytest.raises(SciPlex3V5ObjectiveError, match="wrong exact type"):
        SciPlex3V5ActionContextFit(
            object(),  # type: ignore[arg-type]
            -2.0,
            -2.0,
            -1.0,
            -1.0,
            (substep,),
            (block,),
            gradients,
        )
    with pytest.raises(SciPlex3V5ObjectiveError, match="terminal gates"):
        SciPlex3V5ActionContextFit(
            parameters,
            -2.0,
            -2.0,
            -1.0,
            -1.0,
            (substep,),
            (),
            gradients,
        )
    with pytest.raises(SciPlex3V5ObjectiveError, match="witness chain is invalid"):
        SciPlex3V5ActionContextFit(
            parameters,
            -2.0,
            -2.0,
            -1.0,
            -1.0,
            (),
            (block,),
            gradients,
        )

    first = _plate_substep(parameters=parameters, before=-3.0, after=-2.0)
    discontinuous = _plate_substep(parameters=parameters, before=-1.5, after=-1.0)
    with pytest.raises(SciPlex3V5ObjectiveError, match="discontinuous"):
        SciPlex3V5ActionContextFit(
            parameters,
            -3.0,
            -3.0,
            -1.0,
            -1.0,
            (first, discontinuous),
            (_complete_block(before=-3.0, after=-1.0),),
            gradients,
        )

    with pytest.raises(SciPlex3V5ObjectiveError, match="does not bind the final state"):
        SciPlex3V5ActionContextFit(
            parameters,
            -2.0,
            -2.0,
            -0.5,
            -0.5,
            (substep,),
            (_complete_block(before=-2.0, after=-0.5),),
            gradients,
        )
