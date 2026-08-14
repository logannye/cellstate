"""A read-only HTTP surface over the committed GSE274113 slice.

Every number this serves is computed by the shipped path -- ``cellstate.backends.gse274113`` and
``cellstate.evaluation.gse274113_reports`` -- or is a plain observational statistic over the
committed counts whose construction is written out beside it.  The server originates nothing.

It exists for the one thing ``scripts/explore.py`` cannot do: **vary a model parameter and watch a
measurement respond**.  ``BIOLOGY_RANK`` and ``NUISANCE_RANK`` were constants, so producing the
published rank sensitivity meant editing ``fit.py``, which is not something a reader can do while
looking at the number it moves.

⚠️ **A non-default rank does not produce the published measurements.**  Every response that depends
on the fit carries the ranks it was computed at, and the page marks any panel that is off-default,
so a screenshot cannot be mistaken for the number in the model card.

Read-only, single-user, localhost.  No authentication, no persistence, no write path -- the slice is
pinned by SHA-256 and this process only reads it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..backends.gse274113 import load_arm_slice
from ..backends.gse274113.fit import (
    BIOLOGY_RANK,
    NUISANCE_RANK,
    NULL_TARGET,
    PLACEBO_TARGETS,
    ArmSlice,
    FittedFold,
    fit_fold,
)
from ..backends.gse274113.likelihood import posterior

STATIC = Path(__file__).resolve().parent / "static"

# The rank grid the laboratory sweeps.  Bounded well above the default so the flat region either
# side of the cut is visible, and low enough that a full sweep stays under a second per fold.
RANK_GRID = tuple(range(1, 11))

FloatArray = np.ndarray


# --------------------------------------------------------------------------- request parameters

BiologyRank = Annotated[int, Query(ge=1, le=40, description="columns in the biology basis W")]
NuisanceRank = Annotated[int, Query(ge=1, le=40, description="columns in the nuisance basis V")]


# --------------------------------------------------------------------------- response models


class Ranks(BaseModel):
    """The ranks a result was computed at, and whether they are the published ones."""

    biology: int
    nuisance: int
    is_default: bool
    default_biology: int = BIOLOGY_RANK
    default_nuisance: int = NUISANCE_RANK


class GeneLoadingOut(BaseModel):
    symbol: str
    loading: float


class AxisOut(BaseModel):
    name: str
    coordinate: float
    standard_deviation: float
    top_genes: list[GeneLoadingOut]


class ArmOut(BaseModel):
    library: str
    target: str
    ranks: Ranks
    axes: list[AxisOut]
    abstention_required: bool
    reasons: list[str]


class LibraryOut(BaseModel):
    library: str
    day: int
    cells: int
    panel_counts: int
    nt_cells: int


class InventoryOut(BaseModel):
    libraries: list[LibraryOut]
    targets: list[str]
    placebo_targets: list[str]
    null_target: str
    gene_count: int
    arm_count: int
    total_cells: int
    default_ranks: Ranks


class PanelGeneOut(BaseModel):
    symbol: str
    mean_nt_cpm: float
    is_target: bool
    is_expressed: bool


class ArmStateOut(BaseModel):
    """One arm's biology coordinates, with the covariance needed to draw its uncertainty."""

    target: str
    coordinates: list[float]
    covariance: list[list[float]]
    is_null: bool


class LibraryStatesOut(BaseModel):
    library: str
    ranks: Ranks
    axis_names: list[str]
    arms: list[ArmStateOut]


class BasisColumnOut(BaseModel):
    name: str
    top_genes: list[GeneLoadingOut]


class FoldOut(BaseModel):
    library: str
    ranks: Ranks
    fit_library_count: int
    psi_squared: float
    psi_squared_before_clamp: float
    dispersion_is_clamped: bool
    biology: list[BasisColumnOut]
    nuisance: list[BasisColumnOut]
    residual_norm_by_target: dict[str, float]


class SweepRowOut(BaseModel):
    target: str
    distance: float
    distance_lower_bound_sd: float
    nt_cpm: float
    is_expressed: bool


class SweepOut(BaseModel):
    library: str
    ranks: Ranks
    rows: list[SweepRowOut]
    floor: float | None
    floor_targets: list[str]


class RankRowOut(BaseModel):
    target: str
    mean_rank: float
    best: int
    worst: int
    mean_nt_cpm: float
    is_expressed: bool


class RanksTableOut(BaseModel):
    ranks: Ranks
    rows: list[RankRowOut]
    winners: list[dict[str, str]]
    target_count: int


class IntervalOut(BaseModel):
    """An interval, or the explicit absence of one.

    ``None`` rather than NaN: away from the fitted configuration there is no bootstrap and no
    predeclared bound, so there is no interval -- and a field typed ``float`` that serializes to
    ``null`` would be a model disagreeing with its own contract.
    """

    value: float
    lower: float | None
    upper: float | None


class MeasurementOut(BaseModel):
    name: str
    value: float
    interval: IntervalOut
    unit_count: int
    passed: bool
    statement: str


class DecompositionOut(BaseModel):
    nuisance_across_library: float
    biology_across_library: float
    between_target: float


class MeasureOut(BaseModel):
    ranks: Ranks
    bound: float
    measurements: list[MeasurementOut]
    decomposition: DecompositionOut


class RankResponsePointOut(BaseModel):
    rank: int
    s5: float
    nuisance_across_library: float
    biology_across_library: float
    between_target: float


class RankResponseOut(BaseModel):
    axis: str = Field(description="which rank is being swept: 'biology' or 'nuisance'")
    held_at: int = Field(description="the other rank, held fixed")
    points: list[RankResponsePointOut]


class SpectrumSeriesOut(BaseModel):
    name: str
    row_count: int
    singular_values: list[float]
    normalized: list[float]
    s1_over_s0: float
    pc1_variance_share: float


class SpectrumOut(BaseModel):
    series: list[SpectrumSeriesOut]
    biology_rank_default: int = BIOLOGY_RANK


class KnockdownRowOut(BaseModel):
    target: str
    log2_fold_change: float
    nt_cpm: float
    is_expressed: bool
    wrong_signed: bool


class KnockdownOut(BaseModel):
    rows: list[KnockdownRowOut]
    mean_log2_fold_change: float
    wrong_signed: int
    target_count: int
    mean_well_detected: float
    well_detected_count: int


class DayGeneOut(BaseModel):
    symbol: str
    correlation: float


class DayOut(BaseModel):
    days: list[int]
    libraries_by_day: dict[str, list[str]]
    differentiation_distance: float
    placebo_distance: float
    perturbed_distance: float
    differentiation_over_placebo: float
    perturbed_over_placebo: float
    tracking_gene_count: int
    top_genes: list[DayGeneOut]


class BasisAgreementOut(BaseModel):
    ranks: Ranks
    libraries: list[str]
    axis_names: list[str]
    cosine_by_axis: dict[str, list[list[float]]]
    anchor_gene_by_axis: dict[str, list[str]]
    principal_angles: list[list[float]]
    sign_flips_by_axis: dict[str, int]
    pair_count: int


# --------------------------------------------------------------------------- computation


def _ranks(biology: int, nuisance: int) -> Ranks:
    return Ranks(
        biology=biology,
        nuisance=nuisance,
        is_default=(biology == BIOLOGY_RANK and nuisance == NUISANCE_RANK),
    )


@lru_cache(maxsize=1)
def _slice() -> ArmSlice:
    return load_arm_slice()


@lru_cache(maxsize=512)
def _fold(library: str, biology: int, nuisance: int) -> FittedFold:
    slice_data = _slice()
    if library not in slice_data.libraries:
        raise HTTPException(404, f"unknown library {library!r}")
    try:
        return fit_fold(slice_data, library, biology_rank=biology, nuisance_rank=nuisance)
    except ValueError as error:  # the fit's own rank guards, surfaced rather than swallowed
        raise HTTPException(422, str(error)) from error


def _cpm(slice_data: ArmSlice, library: str, target: str) -> FloatArray:
    counts = slice_data.counts[(library, target)].astype(np.float64)
    return np.asarray(1e6 * (counts + 0.5) / (counts.sum() + counts.shape[0] / 2.0))


def _log_composition(slice_data: ArmSlice, library: str, target: str) -> FloatArray:
    composition, _ = slice_data.log_composition(library, target)
    return np.asarray(composition, dtype=np.float64)


def _top_loadings(column: FloatArray, symbols: tuple[str, ...], count: int) -> list[GeneLoadingOut]:
    """Both poles of an axis, because the informative axes here are contrasts."""

    order = np.argsort(-np.abs(column))[:count]
    return [GeneLoadingOut(symbol=symbols[i], loading=float(column[i])) for i in order]


def _posterior_mean_and_covariance(
    fold: FittedFold, slice_data: ArmSlice, library: str, target: str
) -> tuple[FloatArray, FloatArray]:
    composition, depth = slice_data.log_composition(library, target)
    design = fold.design(NULL_TARGET if target.startswith("NT_") else target)
    return posterior(
        composition,
        intercept=fold.intercept,
        design=design,
        prior_precision=fold.prior_precision(),
        observation_variance_diagonal=fold.observation_variance(depth),
    )


@lru_cache(maxsize=64)
def _held_out_biology(
    biology: int, nuisance: int
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    dict[tuple[str, str], tuple[float, ...]],
    dict[tuple[str, str], tuple[float, ...]],
]:
    """Every arm's biology and nuisance coefficients, each from the fold that excluded its library.

    ⚠️ These are pooled across fourteen *different* fitted bases downstream, which is the declared
    limit recorded in the model card.  It is reproduced here rather than silently corrected, because
    correcting it would make this surface disagree with the published measurements.
    """

    slice_data = _slice()
    libraries = slice_data.libraries
    targets = slice_data.targets
    bio: dict[tuple[str, str], tuple[float, ...]] = {}
    nui: dict[tuple[str, str], tuple[float, ...]] = {}
    for library in libraries:
        fold = _fold(library, biology, nuisance)
        width = fold.biology_basis.shape[1]
        depth = fold.nuisance_basis.shape[1]
        for target in targets:
            mean, _ = _posterior_mean_and_covariance(fold, slice_data, library, target)
            bio[(library, target)] = tuple(float(v) for v in mean[:width])
            nui[(library, target)] = tuple(float(v) for v in mean[width : width + depth])
    return libraries, targets, bio, nui


def _decomposition(biology: int, nuisance: int) -> DecompositionOut:
    libraries, targets, bio, nui = _held_out_biology(biology, nuisance)

    def across_library(block: dict[tuple[str, str], tuple[float, ...]]) -> float:
        means = np.stack([np.mean([block[(lib, t)] for t in targets], axis=0) for lib in libraries])
        return float(np.mean(np.var(means, axis=0)))

    per_target = np.stack([np.mean([bio[(lib, t)] for lib in libraries], axis=0) for t in targets])
    return DecompositionOut(
        nuisance_across_library=across_library(nui),
        biology_across_library=across_library(bio),
        between_target=float(np.mean(np.var(per_target, axis=0))),
    )


def _s5_value(biology: int, nuisance: int) -> float:
    """S5's point estimate at these ranks, without the bootstrap -- for the sweep curve."""

    libraries, targets, bio, _ = _held_out_biology(biology, nuisance)
    means = {t: np.mean([bio[(lib, t)] for lib in libraries], axis=0) for t in targets}
    across_target = float(np.mean(np.var(np.stack(list(means.values())), axis=0)))
    if across_target <= 0.0:
        return float("nan")
    return float(
        np.mean(
            [
                float(np.mean((np.asarray(bio[(lib, t)]) - means[t]) ** 2) / across_target)
                for lib in libraries
                for t in targets
            ]
        )
    )


# --------------------------------------------------------------------------- app

app = FastAPI(
    title="cellstate — GSE274113 explorer",
    description=__doc__,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


@app.get("/api/inventory", response_model=InventoryOut)
def inventory() -> InventoryOut:
    """What the committed slice contains."""

    slice_data = _slice()
    libraries = []
    total = 0
    for library in slice_data.libraries:
        cells = sum(
            count
            for (lib, target), count in slice_data.cells.items()
            if lib == library and target in slice_data.targets
        )
        counts = sum(
            int(slice_data.counts[(lib, target)].sum())
            for (lib, target) in slice_data.counts
            if lib == library and target in slice_data.targets
        )
        total += cells
        libraries.append(
            LibraryOut(
                library=library,
                day=slice_data.library_day[library],
                cells=cells,
                panel_counts=counts,
                nt_cells=slice_data.cells[(library, NULL_TARGET)],
            )
        )
    return InventoryOut(
        libraries=libraries,
        targets=list(slice_data.targets),
        placebo_targets=list(PLACEBO_TARGETS),
        null_target=NULL_TARGET,
        gene_count=len(slice_data.gene_symbols),
        arm_count=len(slice_data.libraries) * len(slice_data.targets),
        total_cells=total,
        default_ranks=_ranks(BIOLOGY_RANK, NUISANCE_RANK),
    )


@app.get("/api/panel", response_model=list[PanelGeneOut])
def panel() -> list[PanelGeneOut]:
    """The 100-gene panel by mean NT expression, with targets and silent genes flagged."""

    slice_data = _slice()
    matrix = np.stack([_cpm(slice_data, lib, NULL_TARGET) for lib in slice_data.libraries])
    mean_cpm = matrix.mean(axis=0)
    targets = set(slice_data.targets)
    order = np.argsort(-mean_cpm)
    return [
        PanelGeneOut(
            symbol=slice_data.gene_symbols[i],
            mean_nt_cpm=float(mean_cpm[i]),
            is_target=slice_data.gene_symbols[i] in targets,
            is_expressed=bool(mean_cpm[i] >= 10.0),
        )
        for i in order
    ]


@app.get("/api/arm/{library}/{target}", response_model=ArmOut)
def arm(
    library: str,
    target: str,
    biology_rank: BiologyRank = BIOLOGY_RANK,
    nuisance_rank: NuisanceRank = NUISANCE_RANK,
    top_genes: Annotated[int, Query(ge=1, le=20)] = 8,
) -> ArmOut:
    """One arm's belief, each axis named by the genes that define it.

    Loadings, never labels.  The abstention is carried through rather than smoothed over: every
    belief this backend emits abstains, and the reasons travel with the coordinates.
    """

    slice_data = _slice()
    if target not in slice_data.targets:
        raise HTTPException(404, f"unknown target {target!r}")
    fold = _fold(library, biology_rank, nuisance_rank)
    mean, covariance = _posterior_mean_and_covariance(fold, slice_data, library, target)
    width = fold.biology_basis.shape[1]
    axes = [
        AxisOut(
            name=f"biology_{i}",
            coordinate=float(mean[i]),
            standard_deviation=float(np.sqrt(max(float(covariance[i][i]), 0.0))),
            top_genes=_top_loadings(fold.biology_basis[:, i], slice_data.gene_symbols, top_genes),
        )
        for i in range(width)
    ]
    # The readiness block is a property of the query, not of the rank, so it is read from the
    # shipped estimator's own verdict rather than restated here.
    from ..backends.gse274113 import describe_state, estimate_arm

    if biology_rank == BIOLOGY_RANK and nuisance_rank == NUISANCE_RANK:
        described = describe_state(estimate_arm(library, target))
        abstention, reasons = described.abstention_required, list(described.reasons)
    else:
        abstention, reasons = (
            True,
            [
                f"computed at biology_rank={biology_rank}, "
                f"nuisance_rank={nuisance_rank}, which is not the fitted configuration "
                "any published measurement was made at",
            ],
        )
    return ArmOut(
        library=library,
        target=target,
        ranks=_ranks(biology_rank, nuisance_rank),
        axes=axes,
        abstention_required=abstention,
        reasons=reasons,
    )


@app.get("/api/library/{library}/states", response_model=LibraryStatesOut)
def library_states(
    library: str,
    biology_rank: BiologyRank = BIOLOGY_RANK,
    nuisance_rank: NuisanceRank = NUISANCE_RANK,
) -> LibraryStatesOut:
    """Every arm of one library in its shared biology coordinates, with covariances.

    One library only.  A belief about library *L* comes from the fold that excluded *L*, so arms in
    different libraries live in different fitted bases and their coordinates do not compare.
    """

    slice_data = _slice()
    fold = _fold(library, biology_rank, nuisance_rank)
    width = fold.biology_basis.shape[1]
    arms = []
    for target in slice_data.targets:
        mean, covariance = _posterior_mean_and_covariance(fold, slice_data, library, target)
        arms.append(
            ArmStateOut(
                target=target,
                coordinates=[float(v) for v in mean[:width]],
                covariance=[[float(covariance[i][j]) for j in range(width)] for i in range(width)],
                is_null=(target == NULL_TARGET),
            )
        )
    return LibraryStatesOut(
        library=library,
        ranks=_ranks(biology_rank, nuisance_rank),
        axis_names=[f"biology_{i}" for i in range(width)],
        arms=arms,
    )


@app.get("/api/fold/{library}", response_model=FoldOut)
def fold_detail(
    library: str,
    biology_rank: BiologyRank = BIOLOGY_RANK,
    nuisance_rank: NuisanceRank = NUISANCE_RANK,
    top_genes: Annotated[int, Query(ge=1, le=20)] = 8,
) -> FoldOut:
    """The fitted W and V of one fold, its dispersion, and each target's residual norm."""

    slice_data = _slice()
    fold = _fold(library, biology_rank, nuisance_rank)
    symbols = slice_data.gene_symbols
    return FoldOut(
        library=library,
        ranks=_ranks(biology_rank, nuisance_rank),
        fit_library_count=len(fold.fit_library_ids),
        psi_squared=fold.biological_observation_variance,
        psi_squared_before_clamp=fold.biological_observation_variance_before_clamp,
        dispersion_is_clamped=fold.dispersion_is_clamped,
        biology=[
            BasisColumnOut(
                name=f"biology_{i}",
                top_genes=_top_loadings(fold.biology_basis[:, i], symbols, top_genes),
            )
            for i in range(fold.biology_basis.shape[1])
        ],
        nuisance=[
            BasisColumnOut(
                name=f"nuisance_{i}",
                top_genes=_top_loadings(fold.nuisance_basis[:, i], symbols, top_genes),
            )
            for i in range(fold.nuisance_basis.shape[1])
        ],
        residual_norm_by_target=dict(fold.residual_norm_by_target),
    )


def _contrast(
    library: str, target_a: str, target_b: str, biology: int, nuisance: int
) -> tuple[float, float]:
    slice_data = _slice()
    fold = _fold(library, biology, nuisance)
    width = fold.biology_basis.shape[1]
    first_mean, first_cov = _posterior_mean_and_covariance(fold, slice_data, library, target_a)
    second_mean, second_cov = _posterior_mean_and_covariance(fold, slice_data, library, target_b)
    difference = np.asarray(second_mean[:width]) - np.asarray(first_mean[:width])
    variance = sum(float(first_cov[i][i]) + float(second_cov[i][i]) for i in range(width))
    return float(np.linalg.norm(difference)), float(np.sqrt(variance))


@app.get("/api/sweep/{library}", response_model=SweepOut)
def sweep(
    library: str,
    biology_rank: BiologyRank = BIOLOGY_RANK,
    nuisance_rank: NuisanceRank = NUISANCE_RANK,
) -> SweepOut:
    """Every target against NT in one library, read against the not-expressed floor.

    ⚠️ The spread reported per row is a **lower** bound: it adds two posterior variances as though
    the arms were independent, and they are not -- both were scored under the same fold.
    """

    slice_data = _slice()
    _fold(library, biology_rank, nuisance_rank)  # 404/422 before any work
    index = {s: i for i, s in enumerate(slice_data.gene_symbols)}
    nt_cpm = _cpm(slice_data, library, NULL_TARGET)
    rows: list[SweepRowOut] = []
    for target in slice_data.targets:
        if target == NULL_TARGET:
            continue
        distance, lower = _contrast(library, NULL_TARGET, target, biology_rank, nuisance_rank)
        expression = float(nt_cpm[index[target]]) if target in index else float("nan")
        rows.append(
            SweepRowOut(
                target=target,
                distance=distance,
                distance_lower_bound_sd=lower,
                nt_cpm=expression,
                is_expressed=bool(expression >= 10.0),
            )
        )
    rows.sort(key=lambda r: -r.distance)
    silent = [r.target for r in rows if not r.is_expressed]
    floor = float(np.mean([r.distance for r in rows if not r.is_expressed])) if silent else None
    return SweepOut(
        library=library,
        ranks=_ranks(biology_rank, nuisance_rank),
        rows=rows,
        floor=floor,
        floor_targets=silent,
    )


@app.get("/api/ranks", response_model=RanksTableOut)
def ranks_table(
    biology_rank: BiologyRank = BIOLOGY_RANK,
    nuisance_rank: NuisanceRank = NUISANCE_RANK,
) -> RanksTableOut:
    """Does the per-library ordering replicate?  Rank every target in all fourteen libraries."""

    slice_data = _slice()
    index = {s: i for i, s in enumerate(slice_data.gene_symbols)}
    perturbed = [t for t in slice_data.targets if t != NULL_TARGET]
    expression = {
        t: float(
            np.mean([_cpm(slice_data, lib, NULL_TARGET)[index[t]] for lib in slice_data.libraries])
        )
        if t in index
        else float("nan")
        for t in perturbed
    }
    positions: dict[str, list[int]] = {t: [] for t in perturbed}
    winners: list[dict[str, str]] = []
    for library in slice_data.libraries:
        distances = {
            t: _contrast(library, NULL_TARGET, t, biology_rank, nuisance_rank)[0] for t in perturbed
        }
        order = sorted(perturbed, key=lambda t: -distances[t])
        for position, target in enumerate(order):
            positions[target].append(position + 1)
        winners.append(
            {
                "library": library,
                "target": order[0],
                "distance": f"{distances[order[0]]:.3f}",
                "is_expressed": str(expression[order[0]] >= 10.0),
            }
        )
    rows = [
        RankRowOut(
            target=t,
            mean_rank=float(np.mean(positions[t])),
            best=min(positions[t]),
            worst=max(positions[t]),
            mean_nt_cpm=expression[t],
            is_expressed=bool(expression[t] >= 10.0),
        )
        for t in sorted(perturbed, key=lambda t: float(np.mean(positions[t])))
    ]
    return RanksTableOut(
        ranks=_ranks(biology_rank, nuisance_rank),
        rows=rows,
        winners=winners,
        target_count=len(perturbed),
    )


@app.get("/api/measure", response_model=MeasureOut)
def measure(
    biology_rank: BiologyRank = BIOLOGY_RANK,
    nuisance_rank: NuisanceRank = NUISANCE_RANK,
    bound: Annotated[float, Query(gt=0.0)] = 0.35,
) -> MeasureOut:
    """S2, S4 both halves, S5, and the three-term block decomposition.

    At the default ranks these are the shipped functions unmodified, so the values are the published
    ones.  Away from the defaults the S2 and S4 measurements are omitted: they are defined by merged
    ADRs at the fitted configuration, and recomputing them elsewhere would present a number that no
    decision authorizes.
    """

    slice_data = _slice()
    measurements: list[MeasurementOut] = []
    if biology_rank == BIOLOGY_RANK and nuisance_rank == NUISANCE_RANK:
        from ..evaluation.gse274113_reports import (
            held_out_states,
            measure_earned_spread,
            measure_intervention_response,
            measure_nuisance_separation,
        )

        states = held_out_states(slice_data)
        null_half, non_null = measure_intervention_response(slice_data)
        for measurement in (
            measure_earned_spread(slice_data),
            null_half,
            non_null,
            measure_nuisance_separation(states, bound=bound),
        ):
            measurements.append(
                MeasurementOut(
                    name=measurement.name,
                    value=measurement.value,
                    interval=IntervalOut(
                        value=measurement.value,
                        lower=measurement.interval.lower,
                        upper=measurement.interval.upper,
                    ),
                    unit_count=measurement.unit_count,
                    passed=measurement.passed,
                    statement=measurement.statement,
                )
            )
    else:
        value = _s5_value(biology_rank, nuisance_rank)
        measurements.append(
            MeasurementOut(
                name="S5 nuisance separation in the inferred state (point estimate only)",
                value=value,
                interval=IntervalOut(value=value, lower=None, upper=None),
                unit_count=len(slice_data.libraries),
                passed=False,
                statement=(
                    f"computed at biology_rank={biology_rank}, "
                    f"nuisance_rank={nuisance_rank}. "
                    "No interval and no verdict: the bound is predeclared for the fitted "
                    "configuration, and this is not it."
                ),
            )
        )
    return MeasureOut(
        ranks=_ranks(biology_rank, nuisance_rank),
        bound=bound,
        measurements=measurements,
        decomposition=_decomposition(biology_rank, nuisance_rank),
    )


@app.get("/api/rank-response", response_model=RankResponseOut)
def rank_response(
    axis: Annotated[str, Query(pattern="^(biology|nuisance)$")] = "biology",
    held_at: Annotated[int, Query(ge=1, le=40)] | None = None,
) -> RankResponseOut:
    """S5 and both component terms across the rank grid, the other rank held fixed.

    This is the shape the published sensitivity (16.96 / 19.22 / 27.21 at ranks 3 / 4 / 5) is three
    points of.  Seeing the whole curve is the point: the cut lands in a flat region.
    """

    other = (
        held_at if held_at is not None else (NUISANCE_RANK if axis == "biology" else BIOLOGY_RANK)
    )
    points: list[RankResponsePointOut] = []
    for rank in RANK_GRID:
        biology, nuisance = (rank, other) if axis == "biology" else (other, rank)
        try:
            decomposition = _decomposition(biology, nuisance)
            points.append(
                RankResponsePointOut(
                    rank=rank,
                    s5=_s5_value(biology, nuisance),
                    nuisance_across_library=decomposition.nuisance_across_library,
                    biology_across_library=decomposition.biology_across_library,
                    between_target=decomposition.between_target,
                )
            )
        except HTTPException:
            continue  # a rank the fit refuses is simply absent from the curve
    return RankResponseOut(axis=axis, held_at=other, points=points)


@app.get("/api/spectrum", response_model=SpectrumOut)
def spectrum() -> SpectrumOut:
    """Is there structure for the biology basis to find?

    The matrix ``W`` is fitted on, against two references from this same slice: the placebo contrast
    (noise by construction) and the differentiation contrast (real biology).
    """

    slice_data = _slice()
    libraries = slice_data.libraries
    composition = {
        (lib, t): _log_composition(slice_data, lib, t)
        for lib in libraries
        for t in (*slice_data.targets, *PLACEBO_TARGETS)
    }
    centred = np.stack([composition[(lib, NULL_TARGET)] for lib in libraries])
    definitions = [
        (
            "perturbation: target - NT",
            [
                composition[(lib, t)] - composition[(lib, NULL_TARGET)]
                for lib in libraries
                for t in slice_data.targets
                if t != NULL_TARGET
            ],
        ),
        (
            "placebo: NT_B - NT_A",
            [
                composition[(lib, PLACEBO_TARGETS[1])] - composition[(lib, PLACEBO_TARGETS[0])]
                for lib in libraries
            ],
        ),
        ("differentiation: NT across days", list(centred - centred.mean(axis=0))),
    ]
    series = []
    for name, rows in definitions:
        values = np.linalg.svd(np.vstack(rows), compute_uv=False)
        share = values**2 / float(np.sum(values**2))
        series.append(
            SpectrumSeriesOut(
                name=name,
                row_count=len(rows),
                singular_values=[float(v) for v in values[:10]],
                normalized=[float(v / values[0]) for v in values[:10]],
                s1_over_s0=float(values[1] / values[0]),
                pc1_variance_share=float(share[0]),
            )
        )
    return SpectrumOut(series=series)


@app.get("/api/knockdown", response_model=KnockdownOut)
def knockdown() -> KnockdownOut:
    """Did the perturbation reach the readout?  A plain statistic over the committed counts."""

    slice_data = _slice()
    index = {s: i for i, s in enumerate(slice_data.gene_symbols)}
    rows: list[KnockdownRowOut] = []
    for target in slice_data.targets:
        if target == NULL_TARGET or target not in index:
            continue
        position = index[target]
        changes = [
            float(
                np.log2(
                    _cpm(slice_data, lib, target)[position]
                    / _cpm(slice_data, lib, NULL_TARGET)[position]
                )
            )
            for lib in slice_data.libraries
        ]
        baseline = float(
            np.mean([_cpm(slice_data, lib, NULL_TARGET)[position] for lib in slice_data.libraries])
        )
        change = float(np.mean(changes))
        rows.append(
            KnockdownRowOut(
                target=target,
                log2_fold_change=change,
                nt_cpm=baseline,
                is_expressed=bool(baseline >= 10.0),
                wrong_signed=bool(change > 0.0),
            )
        )
    rows.sort(key=lambda r: r.log2_fold_change)
    well = [r for r in rows if r.nt_cpm > 200.0]
    return KnockdownOut(
        rows=rows,
        mean_log2_fold_change=float(np.mean([r.log2_fold_change for r in rows])),
        wrong_signed=sum(1 for r in rows if r.wrong_signed),
        target_count=len(rows),
        mean_well_detected=float(np.mean([r.log2_fold_change for r in well])),
        well_detected_count=len(well),
    )


@app.get("/api/day", response_model=DayOut)
def day() -> DayOut:
    """The differentiation readout: raw log-composition, no fitted basis, no regrouping.

    ⚠️ ``library_day`` is nested inside ``library`` -- three or four libraries per day, none
    spanning two -- so this is a readout and deliberately not a measurement.  It carries no interval
    and clears no gate.
    """

    slice_data = _slice()
    by_day: dict[int, list[str]] = {}
    for library in slice_data.libraries:
        by_day.setdefault(slice_data.library_day[library], []).append(library)
    days = sorted(by_day)

    def mean_nt(value: int) -> FloatArray:
        return np.asarray(
            np.mean(
                [_log_composition(slice_data, lib, NULL_TARGET) for lib in by_day[value]], axis=0
            )
        )

    differentiation = float(np.linalg.norm(mean_nt(days[-1]) - mean_nt(days[0])))
    placebo = float(
        np.mean(
            [
                np.linalg.norm(
                    _log_composition(slice_data, lib, PLACEBO_TARGETS[1])
                    - _log_composition(slice_data, lib, PLACEBO_TARGETS[0])
                )
                for lib in slice_data.libraries
            ]
        )
    )
    perturbed = float(
        np.mean(
            [
                np.linalg.norm(
                    _log_composition(slice_data, lib, t)
                    - _log_composition(slice_data, lib, NULL_TARGET)
                )
                for lib in slice_data.libraries
                for t in slice_data.targets
                if t != NULL_TARGET
            ]
        )
    )
    day_values = np.array([slice_data.library_day[lib] for lib in slice_data.libraries], float)
    matrix = np.stack(
        [_log_composition(slice_data, lib, NULL_TARGET) for lib in slice_data.libraries]
    )
    correlation = np.nan_to_num(
        np.array([np.corrcoef(day_values, matrix[:, g])[0, 1] for g in range(matrix.shape[1])])
    )
    order = np.argsort(-np.abs(correlation))[:12]
    return DayOut(
        days=days,
        libraries_by_day={str(d): by_day[d] for d in days},
        differentiation_distance=differentiation,
        placebo_distance=placebo,
        perturbed_distance=perturbed,
        differentiation_over_placebo=differentiation / placebo,
        perturbed_over_placebo=perturbed / placebo,
        tracking_gene_count=int((np.abs(correlation) > 0.7).sum()),
        top_genes=[
            DayGeneOut(symbol=slice_data.gene_symbols[i], correlation=float(correlation[i]))
            for i in order
        ],
    )


@app.get("/api/basis", response_model=BasisAgreementOut)
def basis_agreement(
    biology_rank: BiologyRank = BIOLOGY_RANK,
    nuisance_rank: NuisanceRank = NUISANCE_RANK,
) -> BasisAgreementOut:
    """Do the fourteen fitted bases agree about which direction each axis points?

    This is the one place fold bases are put side by side, and its entire purpose is to show that
    they disagree -- the declared limit under which S5 and the block decomposition are computed.
    """

    slice_data = _slice()
    libraries = list(slice_data.libraries)
    bases = {lib: _fold(lib, biology_rank, nuisance_rank).biology_basis for lib in libraries}
    width = bases[libraries[0]].shape[1]
    names = [f"biology_{i}" for i in range(width)]

    cosine: dict[str, list[list[float]]] = {}
    flips: dict[str, int] = {}
    anchors: dict[str, list[str]] = {}
    for axis in range(width):
        matrix = [
            [float(bases[a][:, axis] @ bases[b][:, axis]) for b in libraries] for a in libraries
        ]
        cosine[names[axis]] = matrix
        flips[names[axis]] = sum(
            1
            for i in range(len(libraries))
            for j in range(i + 1, len(libraries))
            if matrix[i][j] < 0.0
        )
        anchors[names[axis]] = [
            slice_data.gene_symbols[int(np.argmax(np.abs(bases[lib][:, axis])))]
            for lib in libraries
        ]

    angles = [
        [
            float(
                np.degrees(
                    np.arccos(
                        np.clip(np.linalg.svd(bases[a].T @ bases[b], compute_uv=False), -1.0, 1.0)
                    )
                ).max()
            )
            for b in libraries
        ]
        for a in libraries
    ]
    pairs = len(libraries) * (len(libraries) - 1) // 2
    return BasisAgreementOut(
        ranks=_ranks(biology_rank, nuisance_rank),
        libraries=libraries,
        axis_names=names,
        cosine_by_axis=cosine,
        anchor_gene_by_axis=anchors,
        principal_angles=angles,
        sign_flips_by_axis=flips,
        pair_count=pairs,
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    """Serve on localhost.

    Bound to 127.0.0.1 deliberately: this reads a research slice and is not a public service.
    """

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
