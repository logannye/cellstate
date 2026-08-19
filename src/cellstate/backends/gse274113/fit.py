"""Leave-one-library-out fits of the GSE274113 RNA observation model.

The library is the independent experimental unit (program rule 8), so a belief about an arm in
library L may only come from a fit that never saw L.  Fourteen libraries give fourteen folds, and
the estimator refuses an arm whose library is inside its own fit set -- a refusal with a reachable
false branch, not a comment.

**Where the three subspaces come from, and why that is the S5 claim.**  S5 asks that varying a
nuisance variable at fixed biology move the predicted observation and not the inferred state.  The
design encodes that split structurally rather than hoping the fit discovers it:

* ``V`` -- the nuisance basis -- is the leading subspace of the NT arms' residuals across
  libraries.  ⚠️ An earlier revision of this docstring justified that by asserting "NT is the same
  biology in every library, so whatever moves there is the library."  **That premise is false in
  this deposit and is withdrawn.**  The fourteen libraries sit at four differentiation days, so NT
  at day 7 and NT at day 14 are not the same biology, and the across-library NT residual carries
  the culture's differentiation clock.  Measured rather than argued: ``V`` absorbs **0.999** of the
  day-7-to-day-14 NT direction in fourteen of fourteen folds, against about 0.03 for a random
  three-dimensional subspace and under 0.15 for the placebo contrast, which is sampling noise by
  construction.  Every fold now reports the number as ``day_axis_in_nuisance_basis``, so the claim
  is computed instead of asserted.
* ``W`` -- the biology basis -- is the leading subspace of the *within-library* contrasts
  ``c[L, g] - c[L, NT]``.  Differencing inside a library cancels the library, so whatever moves
  there is the perturbation.
* ``u_g`` -- one direction per target -- is what that target does beyond the shared basis.

``W`` is then orthogonalized against ``V``.  **That is a declared bias, not a neutrality**: any
biology genuinely aligned with the library axis is assigned to nuisance, which biases against
finding biology rather than for it.  Stated here because a choice this consequential should not
have to be reverse-engineered from the code.

What the measurement above adds to that declaration is its size.  Because ``V`` holds 0.999 of the
day axis and ``W`` is orthogonalized against ``V``, the biology block is left with essentially none
of the strongest biological signal this deposit contains -- ``day_axis_in_biology_basis`` is under
0.01.  **So what this backend calls biology is, by construction, the residue after differentiation
is set aside.**  Whether that is the right split is a question about the estimand and not about this
function; it is not repaired here, and repairing it would need its own decision record.

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

# Floor on the fitted extra-multinomial dispersion.  Reaching it is a real signal about a fold, so
# ``FittedFold`` records the pre-clamp value and ``dispersion_is_clamped`` reports it (ADR 0022
# decision 4).  A silent floor is what let "psi^2 is fitted" stay false for fourteen folds.
DISPERSION_FLOOR = 1e-6

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
    pooled_rate: FloatArray
    biology_prior_variance: float
    nuisance_prior_variance: float
    realization_prior_variance: float
    biological_observation_variance: float
    biological_observation_variance_before_clamp: float
    residual_norm_by_target: dict[str, float]
    # Measured per fold rather than asserted. `fit.py`'s stated premise -- "NT is the same biology
    # in every library, so whatever moves there is the library" -- is false in this deposit: the
    # fourteen libraries sit at four differentiation days, so the across-library NT residual that
    # `V` is fitted on contains the culture's differentiation clock. `None` when the fit libraries
    # span a single day, because then there is no day direction to project and a 0.0 would be
    # indistinguishable from a fold that genuinely absorbs nothing.
    day_axis_in_nuisance_basis: float | None
    day_axis_in_biology_basis: float | None

    @property
    def dispersion_is_clamped(self) -> bool:
        """Whether ``psi^2`` reached its floor rather than being fitted.

        Under ADR 0022 this is measured per fold and reported, never assumed to be false.  It was
        true in fourteen of fourteen folds before that decision, which is what made
        ``likelihood.py``'s "it is fitted, not assumed" a false statement about shipped behaviour.
        """

        return self.biological_observation_variance_before_clamp <= DISPERSION_FLOOR

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

    def observation_variance(self, depth: float) -> FloatArray:
        """Total per-gene observation variance for an arm at this depth.

        It no longer takes the arm's counts: under ADR 0022 the technical term is evaluated at the
        fold's pooled rate, so an arm's own observation cannot inflate the variance it is weighted
        by.  Depth still enters, because a deeper arm genuinely carries less sampling noise.
        """

        return observation_variance(
            technical_variance(self.pooled_rate, depth),
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


def _leading_subspace(rows: FloatArray, rank: int, *, name: str) -> FloatArray:
    """The ``rank`` leading right-singular vectors of ``rows``.

    ⚠️ ``np.linalg.svd`` returns only ``min(rows, columns)`` vectors, so ``right[:rank]`` on a
    matrix with fewer rows than ``rank`` **silently returns fewer columns than asked for**.  The
    fold would then report a rank nobody requested, which is a request that appears to succeed
    while doing something else.  Refused instead.
    """

    available = min(rows.shape[0], rows.shape[1])
    if rank > available:
        raise ValueError(
            f"{name}_rank={rank} exceeds the {available} directions this fold can resolve "
            f"({rows.shape[0]} rows x {rows.shape[1]} genes); a larger rank would silently "
            "return a smaller basis"
        )
    _, _, right = np.linalg.svd(rows, full_matrices=False)
    return _canonical_signs(np.asarray(right[:rank].T, dtype=np.float64))


def _orthonormalize_against(basis: FloatArray, against: FloatArray) -> FloatArray:
    projected = basis - against @ (against.T @ basis)
    orthonormal, _ = np.linalg.qr(projected)
    return _canonical_signs(np.asarray(orthonormal[:, : basis.shape[1]], dtype=np.float64))


def _day_axis_absorption(
    slice_data: ArmSlice,
    fit_libraries: tuple[str, ...],
    nuisance_basis: FloatArray,
    biology_basis: FloatArray,
) -> tuple[float | None, float | None]:
    """Squared projection of the earliest-to-latest ``NT`` direction onto each block.

    Returns ``(None, None)`` when the fit libraries span one day: there is no day contrast to
    project, and a zero would read as "absorbs nothing" rather than "not defined here".

    The direction is taken between the mean ``NT`` composition at the earliest and latest day
    present, which is a plain observational contrast -- no fitted basis, no model.  A random
    three-dimensional subspace of a hundred-gene space would capture about 0.03.
    """

    by_day: dict[int, list[FloatArray]] = {}
    for library in fit_libraries:
        composition, _ = slice_data.log_composition(library, NULL_TARGET)
        by_day.setdefault(slice_data.library_day[library], []).append(composition)
    if len(by_day) < 2:
        return None, None

    days = sorted(by_day)
    direction = np.mean(by_day[days[-1]], axis=0) - np.mean(by_day[days[0]], axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return None, None
    unit = np.asarray(direction / norm, dtype=np.float64)
    return (
        float(np.sum((nuisance_basis.T @ unit) ** 2)),
        float(np.sum((biology_basis.T @ unit) ** 2)),
    )


def fit_fold(
    slice_data: ArmSlice,
    held_out_library: str,
    *,
    biology_rank: int = BIOLOGY_RANK,
    nuisance_rank: int = NUISANCE_RANK,
) -> FittedFold:
    """Fit the observation model on every library except ``held_out_library``.

    ``biology_rank`` and ``nuisance_rank`` default to the module constants, so every existing
    caller and every pinned measurement is unaffected.  They are arguments rather than constants
    because **the rank is a free parameter and reads as one**: the biology contrast matrix's
    singular values fall 40.78, 30.59, 25.78, 24.34, 21.49, so a cut at four separates values that
    differ by 12%.  Sweeping the rank was previously an edit to this file, which is not a thing a
    reader can do while looking at the number it moves.

    ⚠️ Non-default ranks do not produce the published measurements, and nothing downstream
    renumbers itself to say so.  A caller that varies them owns saying which rank it used.
    """

    if held_out_library not in slice_data.libraries:
        raise ValueError(f"unknown library {held_out_library!r}")
    panel_size = len(slice_data.gene_symbols)
    if biology_rank < 1 or nuisance_rank < 1:
        raise ValueError(
            f"both ranks must be at least one; got biology_rank={biology_rank}, "
            f"nuisance_rank={nuisance_rank}"
        )
    # The shared basis [W | V] plus one target direction must stay within the panel, or the design
    # is rank-deficient and the posterior is not identified.
    if biology_rank + nuisance_rank >= panel_size:
        raise ValueError(
            f"biology_rank + nuisance_rank must be under the {panel_size}-gene panel width; "
            f"got {biology_rank} + {nuisance_rank}"
        )
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

    # The rate the technical variance is evaluated at (ADR 0022).  Pooled over the fit libraries
    # only, so a held-out library contributes nothing to the variance its own arms are weighted by.
    # Pooling per target was measured and agrees to four decimal places on the fitted dispersion, so
    # the flatter scope is taken and one leakage surface is avoided.
    #
    # The rate is pooled over the **Haldane-corrected** compositions, the same statistic the model
    # is written on.  Raw proportions would estimate a never-observed gene's rate as exactly zero,
    # and a zero rate is not a small rate: it makes the sampling variance zero, hence the precision
    # infinite, and the posterior would weight that gene without bound.  Two panel genes are at or
    # near zero throughout this slice, so this is a reachable case rather than a hypothetical.
    pooled_rate = np.mean(
        [np.exp(compositions[(lib, t)]) for lib in fit_libraries for t in slice_data.targets],
        axis=0,
    )

    # Nuisance: NT is the same biology everywhere, so its across-library spread IS the library.
    nuisance_rows = np.vstack(
        [compositions[(lib, NULL_TARGET)] - intercept for lib in fit_libraries]
    )
    nuisance_basis = _leading_subspace(nuisance_rows, nuisance_rank, name="nuisance")

    # Biology: differencing within a library cancels the library.
    contrasts = {
        (lib, t): compositions[(lib, t)] - compositions[(lib, NULL_TARGET)]
        for lib in fit_libraries
        for t in perturbed
    }
    biology_rows = np.vstack(list(contrasts.values()))
    biology_basis = _orthonormalize_against(
        _leading_subspace(biology_rows, biology_rank, name="biology"), nuisance_basis
    )

    # The bases must be exactly the requested width.  Anything else is a fold misreporting its
    # own rank, and every consumer reads the rank off `basis.shape[1]`.
    assert biology_basis.shape[1] == biology_rank, "biology basis is not the requested rank"
    assert nuisance_basis.shape[1] == nuisance_rank, "nuisance basis is not the requested rank"

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
    # This subtraction is only meaningful if the term being subtracted is right, and for fourteen
    # folds it was not: evaluated at the arm's own counts the technical term overstated the zero
    # bucket 6.4x, drove the difference below negative 0.16, and pinned the result at the floor
    # (ADR 0022).  The pre-clamp value is carried on the fold now, so a return to that state is
    # visible rather than silent.
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
            technical_means.append(float(np.mean(technical_variance(pooled_rate, depths[key]))))
    raw_variance = float(np.mean(residual_squares) - np.mean(technical_means))
    biological_variance = max(raw_variance, DISPERSION_FLOOR)

    # How much of the differentiation axis each block absorbs.  `W` is orthogonalized against `V`
    # above, so whatever `V` takes is removed from biology by construction; reporting both numbers
    # is what lets a reader see that rather than infer it.
    day_in_nuisance, day_in_biology = _day_axis_absorption(
        slice_data, fit_libraries, nuisance_basis, biology_basis
    )

    return FittedFold(
        held_out_library=held_out_library,
        fit_library_ids=fit_libraries,
        intercept=np.asarray(intercept, dtype=np.float64),
        biology_basis=biology_basis,
        nuisance_basis=nuisance_basis,
        target_directions=target_directions,
        pooled_rate=np.asarray(pooled_rate, dtype=np.float64),
        biology_prior_variance=biology_prior,
        nuisance_prior_variance=nuisance_prior,
        realization_prior_variance=realization_prior,
        biological_observation_variance=biological_variance,
        biological_observation_variance_before_clamp=raw_variance,
        residual_norm_by_target=residual_norm_by_target,
        day_axis_in_nuisance_basis=day_in_nuisance,
        day_axis_in_biology_basis=day_in_biology,
    )
