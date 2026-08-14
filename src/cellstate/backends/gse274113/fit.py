"""Leave-one-library-out fits of the GSE274113 RNA observation model.

The library is the independent experimental unit (program rule 8), so a belief about an arm in
library L may only come from a fit that never saw L.  Fourteen libraries give fourteen folds, and
the estimator refuses an arm whose library is inside its own fit set -- a refusal with a reachable
false branch, not a comment.

**Where the three subspaces come from, and why that is the S5 claim.**  S5 asks that varying a
nuisance variable at fixed biology move the predicted observation and not the inferred state.  The
design encodes that split structurally rather than hoping the fit discovers it:

* ``V`` -- the nuisance basis -- is the leading subspace of the NT arms' residuals across
  libraries.  NT is the same biology in every library, so whatever moves there is the library.
* ``W`` -- the biology basis -- is the leading subspace of the *within-library* contrasts
  ``c[L, g] - c[L, NT]``.  Differencing inside a library cancels the library, so whatever moves
  there is the perturbation.
* ``u_g`` -- one direction per target -- is what that target does beyond the shared basis.

``W`` is then orthogonalized against ``V``.  **That is a declared bias, not a neutrality**: any
biology genuinely aligned with the library axis is assigned to nuisance, which biases against
finding biology rather than for it.  Stated here because a choice this consequential should not
have to be reverse-engineered from the code.

**NT gets a real direction, not a structural zero.**  If ``u_NT`` were fixed at zero, S4's null half
-- a declared-null intervention must leave the predictive distribution unchanged -- could not fail,
and ADR 0019 warns in as many words that an inert ``do`` operator passes every test this repository
has.  So NT's direction is estimated from a deterministic within-library placebo split of the NT
cells themselves.  It travels through the identical estimator as every perturbed target, and its
expected magnitude is zero because both halves are the same biology.  The null half is therefore a
measurement that can come out wrong.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .likelihood import log_composition, observation_variance, technical_variance

FloatArray = NDArray[np.float64]

BIOLOGY_RANK = 4
NUISANCE_RANK = 3
NULL_TARGET = "NT"
PLACEBO_TARGETS = ("NT_A", "NT_B")

__all__ = [
    "BIOLOGY_RANK",
    "NUISANCE_RANK",
    "ArmSlice",
    "FittedFold",
    "fit_fold",
]


@dataclass(frozen=True)
class ArmSlice:
    """The checked-in panel slice: one count vector per ``(library, target)`` arm."""

    gene_symbols: tuple[str, ...]
    libraries: tuple[str, ...]
    targets: tuple[str, ...]
    library_day: dict[str, int]
    counts: dict[tuple[str, str], NDArray[np.int64]]
    cells: dict[tuple[str, str], int]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ArmSlice:
        counts: dict[tuple[str, str], NDArray[np.int64]] = {}
        cells: dict[tuple[str, str], int] = {}
        for arm in payload["arms"]:
            vector = arm["counts"]
            if vector is None:
                # A zero-total arm was never measured; it is absent, never a row of zeros.
                continue
            key = (str(arm["library"]), str(arm["target"]))
            counts[key] = np.asarray(vector, dtype=np.int64)
            cells[key] = int(arm["cells"])
        return cls(
            gene_symbols=tuple(str(value) for value in payload["gene_symbols"]),
            libraries=tuple(str(value) for value in payload["libraries"]),
            targets=tuple(str(value) for value in payload["targets"]),
            library_day={str(key): int(value) for key, value in payload["library_day"].items()},
            counts=counts,
            cells=cells,
        )

    def log_composition(self, library: str, target: str) -> tuple[FloatArray, float]:
        return log_composition(self.counts[(library, target)])


@dataclass(frozen=True)
class FittedFold:
    """Parameters of one leave-one-library-out fit."""

    held_out_library: str
    fit_library_ids: tuple[str, ...]
    intercept: FloatArray
    biology_basis: FloatArray
    nuisance_basis: FloatArray
    target_directions: dict[str, FloatArray]
    biology_prior_variance: float
    nuisance_prior_variance: float
    realization_prior_variance: float
    biological_observation_variance: float
    residual_norm_by_target: dict[str, float]

    @property
    def state_dimension(self) -> int:
        return int(self.biology_basis.shape[1]) + int(self.nuisance_basis.shape[1]) + 1

    def design(self, target: str) -> FloatArray:
        """``[W | V | u_g]`` -- the columns whose coefficients are the latent state."""

        direction = self.target_directions[target][:, None]
        return np.asarray(
            np.hstack([self.biology_basis, self.nuisance_basis, direction]), dtype=np.float64
        )

    def prior_precision(self) -> FloatArray:
        diagonal = np.concatenate(
            [
                np.full(self.biology_basis.shape[1], 1.0 / self.biology_prior_variance),
                np.full(self.nuisance_basis.shape[1], 1.0 / self.nuisance_prior_variance),
                np.array([1.0 / self.realization_prior_variance]),
            ]
        )
        return np.asarray(np.diag(diagonal), dtype=np.float64)

    def observation_variance(self, counts: NDArray[np.int64], depth: float) -> FloatArray:
        return observation_variance(
            technical_variance(counts, depth),
            self.biological_observation_variance,
        )


def _canonical_signs(basis: FloatArray) -> FloatArray:
    """Fix each column's sign by its largest-magnitude entry.

    SVD sign is arbitrary and varies across BLAS builds.  Left unfixed it makes a checked-in fold
    artifact irreproducible on another machine for no scientific reason.
    """

    signs = np.sign(basis[np.argmax(np.abs(basis), axis=0), np.arange(basis.shape[1])])
    signs[signs == 0.0] = 1.0
    return np.asarray(basis * signs, dtype=np.float64)


def _leading_subspace(rows: FloatArray, rank: int) -> FloatArray:
    _, _, right = np.linalg.svd(rows, full_matrices=False)
    return _canonical_signs(np.asarray(right[:rank].T, dtype=np.float64))


def _orthonormalize_against(basis: FloatArray, against: FloatArray) -> FloatArray:
    projected = basis - against @ (against.T @ basis)
    orthonormal, _ = np.linalg.qr(projected)
    return _canonical_signs(np.asarray(orthonormal[:, : basis.shape[1]], dtype=np.float64))


def fit_fold(slice_data: ArmSlice, held_out_library: str) -> FittedFold:
    """Fit the observation model on every library except ``held_out_library``."""

    if held_out_library not in slice_data.libraries:
        raise ValueError(f"unknown library {held_out_library!r}")
    fit_libraries = tuple(lib for lib in slice_data.libraries if lib != held_out_library)
    perturbed = tuple(t for t in slice_data.targets if t != NULL_TARGET)

    compositions: dict[tuple[str, str], FloatArray] = {}
    depths: dict[tuple[str, str], float] = {}
    for library in fit_libraries:
        for target in (*slice_data.targets, *PLACEBO_TARGETS):
            key = (library, target)
            if key in slice_data.counts:
                compositions[key], depths[key] = slice_data.log_composition(library, target)

    intercept = np.mean(
        [compositions[(lib, t)] for lib in fit_libraries for t in slice_data.targets], axis=0
    )

    # Nuisance: NT is the same biology everywhere, so its across-library spread IS the library.
    nuisance_rows = np.vstack(
        [compositions[(lib, NULL_TARGET)] - intercept for lib in fit_libraries]
    )
    nuisance_basis = _leading_subspace(nuisance_rows, NUISANCE_RANK)

    # Biology: differencing within a library cancels the library.
    contrasts = {
        (lib, t): compositions[(lib, t)] - compositions[(lib, NULL_TARGET)]
        for lib in fit_libraries
        for t in perturbed
    }
    biology_rows = np.vstack(list(contrasts.values()))
    biology_basis = _orthonormalize_against(
        _leading_subspace(biology_rows, BIOLOGY_RANK), nuisance_basis
    )

    shared = np.hstack([biology_basis, nuisance_basis])

    def residual_direction(vector: FloatArray) -> tuple[FloatArray, float]:
        residual = vector - shared @ (shared.T @ vector)
        norm = float(np.linalg.norm(residual))
        if norm < 1e-12:
            # Degenerate: the target is fully explained by the shared basis.  Return a zero
            # direction and let the caller record it as weakly identified rather than divide by ~0.
            return np.zeros_like(vector), 0.0
        return np.asarray(residual / norm, dtype=np.float64), norm

    target_directions: dict[str, FloatArray] = {}
    residual_norm_by_target: dict[str, float] = {}
    for target in perturbed:
        mean_contrast = np.mean([contrasts[(lib, target)] for lib in fit_libraries], axis=0)
        direction, norm = residual_direction(np.asarray(mean_contrast, dtype=np.float64))
        target_directions[target] = direction
        residual_norm_by_target[target] = norm

    # NT's direction comes from the placebo split, so the null half is estimated, not assumed.
    placebo_libraries = [
        lib for lib in fit_libraries if all((lib, half) in compositions for half in PLACEBO_TARGETS)
    ]
    if not placebo_libraries:
        raise ValueError("the placebo split is missing; the S4 null half would be inert")
    placebo_contrast = np.mean(
        [
            compositions[(lib, PLACEBO_TARGETS[1])] - compositions[(lib, PLACEBO_TARGETS[0])]
            for lib in placebo_libraries
        ],
        axis=0,
    )
    direction, norm = residual_direction(np.asarray(placebo_contrast, dtype=np.float64))
    target_directions[NULL_TARGET] = direction
    residual_norm_by_target[NULL_TARGET] = norm

    # Prior scales: the empirical spread of each block's projections over the fit arms.
    centered = np.vstack(
        [compositions[(lib, t)] - intercept for lib in fit_libraries for t in slice_data.targets]
    )
    biology_projection = centered @ biology_basis
    nuisance_projection = centered @ nuisance_basis
    realization_projection = np.array(
        [
            float((compositions[(lib, t)] - intercept) @ target_directions[t])
            for lib in fit_libraries
            for t in slice_data.targets
        ]
    )

    def positive_variance(values: FloatArray) -> float:
        return max(float(np.var(values)), 1e-8)

    biology_prior = positive_variance(biology_projection)
    nuisance_prior = positive_variance(nuisance_projection)
    realization_prior = positive_variance(realization_projection)

    # Extra-multinomial dispersion: what the model does not explain, beyond sampling noise.
    #
    # The arm's own nuisance coefficients are fitted, because at inference they are inferred from
    # the arm's own counts -- a held-out library's batch effect is estimated, not predicted.  So the
    # residual measured here is the dispersion that remains once the library is accounted for, and
    # it is compared against the corrected sampling floor.
    #
    # An out-of-sample variant was tried first, predicting each arm from the mean coefficients of
    # its target across libraries.  It is wrong here and the error is instructive: averaging over
    # libraries strips the arm's nuisance coefficients, so the whole library effect reappears in the
    # residual and is charged to observation noise -- while the V columns are *also* modelling it.
    # That double-counting inflated this term to 1.99 and washed every perturbation signal out of
    # the posterior.
    #
    # The consequence of the correct choice is stated rather than hidden: with the sampling floor
    # this small, the posterior is tight, and whether that tightness is *earned* is not something
    # this estimator may assert about itself.  The calibration report adjudicates it, and it can
    # fail.
    residual_squares: list[float] = []
    technical_means: list[float] = []
    degrees_of_freedom = biology_basis.shape[1] + nuisance_basis.shape[1] + 1
    for library in fit_libraries:
        for target in slice_data.targets:
            key = (library, target)
            observed = compositions[key]
            design_t = np.hstack(
                [biology_basis, nuisance_basis, target_directions[target][:, None]]
            )
            coefficients, *_ = np.linalg.lstsq(design_t, observed - intercept, rcond=None)
            residual = observed - intercept - design_t @ coefficients
            panel_size = residual.shape[0]
            scale = panel_size / max(panel_size - degrees_of_freedom, 1)
            residual_squares.append(float(np.mean(residual**2)) * scale)
            technical_means.append(
                float(np.mean(technical_variance(slice_data.counts[key], depths[key])))
            )
    biological_variance = max(float(np.mean(residual_squares) - np.mean(technical_means)), 1e-6)

    return FittedFold(
        held_out_library=held_out_library,
        fit_library_ids=fit_libraries,
        intercept=np.asarray(intercept, dtype=np.float64),
        biology_basis=biology_basis,
        nuisance_basis=nuisance_basis,
        target_directions=target_directions,
        biology_prior_variance=biology_prior,
        nuisance_prior_variance=nuisance_prior,
        realization_prior_variance=realization_prior,
        biological_observation_variance=biological_variance,
        residual_norm_by_target=residual_norm_by_target,
    )
