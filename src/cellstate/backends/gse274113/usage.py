"""One call from the committed slice to a belief, and a readable account of what it says.

Everything this module does was already possible; none of it was reachable without knowing things
that only a test fixture recorded.  Estimating a single arm required constructing the estimator
**twice** -- once with a placeholder fingerprint, to learn the real one, because
:func:`arm_query` needs a ``model_fingerprint`` that only an estimator can compute -- plus two
SHA-256 digests, a slice load, a fold fit, and the unpacking of a composition, a depth and a cell
count into :func:`arm_request`.  That is roughly forty lines to ask one question, and the only
worked example in the repository uses the *non-biological reference* backend instead.

So the functions here add no capability and make no new claim.  They are a surface over quantities
this backend already computes, and they exist because **a representation nobody can look at cannot
be iterated on.**

What the readout is for.  A belief's biology block is four coordinates named ``biology_0`` through
``biology_3``, which is not something a reader can act on.  The basis they are coordinates *in* is a
100-gene loading matrix, so naming each axis by the genes that define it turns an opaque vector into
an interpretable one.  On the committed fold the leading axes come out as recognisable
haematopoietic biology -- primary-granule genes against B-cell and platelet markers -- which is the
representation working, and which nothing in this repository had ever printed.

⚠️ **Loadings are reported; labels are not.**  This module will tell you that ``biology_0`` loads
``MPO +0.33`` and ``CD79A -0.31``.  It will not call that "the granulocyte axis", because that is an
interpretation and this project does not ship interpretations as fields.  Naming it in an
``axis_label`` column would be the same defect as a hardcoded verdict: a claim no measurement backs.

⚠️ **Abstention is carried through, not smoothed over.**  Every belief from this backend abstains,
and :class:`StateDescription` reprints the reasons rather than presenting a coordinate as an answer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ...api import estimate_cell_state
from ...domain.belief import CalibrationReport, CellStateBelief, StateFactor
from ...domain.distributions import ParametricDistribution
from .arm_request import arm_query, arm_request
from .estimator import GSE274113ObservationEstimator
from .fit import ArmSlice, FittedFold, fit_fold

__all__ = [
    "ArmContrast",
    "AxisReadout",
    "GeneLoading",
    "StateDescription",
    "artifact_directory",
    "available_arms",
    "compare_arms",
    "describe_state",
    "estimate_arm",
    "load_arm_slice",
]

_PLACEHOLDER_FINGERPRINT = "0" * 64


def artifact_directory() -> Path:
    """Where the committed panel and slice live.

    Two locations are tried, in this order, because both are real and neither can be assumed:

    1. ``cellstate/backends/gse274113/_slice`` -- where the wheel puts it.  The build
       force-includes the 336 KB slice so that an *installed* ``cellstate`` can answer a question.
       It could not before: the backend imported cleanly and then raised here, which made the
       flagship path dead in every ``--no-editable`` install, including the mode every Makefile
       target uses.
    2. ``<repo>/backends/vertical-a/gse274113-rna-obs-v1`` -- where a source checkout keeps it,
       and the only copy that exists when running straight from ``src/``.

    Resolved rather than assumed, and every entry point below still accepts an explicit directory.
    Failing loudly here beats a confusing error four frames deeper.
    """

    packaged = Path(__file__).resolve().parent / "_slice"
    checkout = (
        Path(__file__).resolve().parents[4] / "backends" / "vertical-a" / "gse274113-rna-obs-v1"
    )
    for directory in (packaged, checkout):
        if (directory / "arms.json").is_file():
            return directory
    raise FileNotFoundError(
        f"the committed GSE274113 slice is at neither {packaged} nor {checkout}; "
        "pass an explicit directory, or run from a source checkout"
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def load_arm_slice(directory: Path | None = None) -> ArmSlice:
    """Load the committed slice.  Cached, because every entry point below wants the same one."""

    resolved = directory or artifact_directory()
    return ArmSlice.from_payload(json.loads((resolved / "arms.json").read_text(encoding="utf-8")))


@lru_cache(maxsize=4)
def _calibration(
    directory: Path, minimum_coverage: float, maximum_calibration_error: float
) -> CalibrationReport:
    """S6 for the whole deposit, computed once and shared by every fold's estimator.

    **It is computed, never stored.**  Fitting all fourteen folds and bootstrapping the coverage
    takes about a fifth of a second on the committed slice, which is cheap enough that there is no
    reason to commit the number as an artifact -- and committing it would recreate the failure mode
    this project keeps finding, where a recorded claim and the code that would check it drift apart
    without anything going red.  There is nothing here to drift: the belief reports whatever the
    slice currently measures.
    """

    # Imported here, not at module scope: `evaluation` measures `backends`, so the module-level
    # import would close a cycle through `backends/gse274113/__init__.py`.  It did, and the whole
    # suite stayed green -- 34 tests passed while `import cellstate.evaluation.gse274113_reports`
    # as a program's FIRST import raised ImportError, because no test ever imported in that order.
    from ...evaluation.gse274113_reports import measure_calibration_coverage

    return measure_calibration_coverage(
        load_arm_slice(directory),
        minimum_coverage=minimum_coverage,
        maximum_calibration_error=maximum_calibration_error,
    )


@lru_cache(maxsize=32)
def _fold_and_estimator(
    library: str, directory: Path | None = None
) -> tuple[FittedFold, GSE274113ObservationEstimator]:
    """The fold that EXCLUDES ``library``, and an estimator bound to it.

    This is the two-pass fingerprint construction, done once and cached.  It is not incidental
    ceremony: the query commits to the model that answers it, so the query cannot be built until
    the model's fingerprint exists, and the fingerprint cannot be computed without a model.  The
    placeholder pass exists to break that cycle, and every caller was previously expected to
    rediscover it.
    """

    resolved = directory or artifact_directory()
    slice_data = load_arm_slice(resolved)
    if library not in slice_data.libraries:
        raise ValueError(
            f"unknown library {library!r}; the slice carries {list(slice_data.libraries)}"
        )
    fold = fit_fold(slice_data, library)
    digests = {
        "slice_fingerprint": _digest(resolved / "arms.json"),
        "panel_fingerprint": _digest(resolved / "panel.json"),
    }
    placeholder = arm_query(slice_data.targets, model_fingerprint=_PLACEHOLDER_FINGERPRINT)
    # The two thresholds are read off the query rather than restated, so the pair the report is
    # scored against cannot drift from the pair the belief validates against.  `CellStateBelief`
    # rejects a mismatch outright, so a restated constant would turn a silent divergence into a
    # loud one -- but reading them here means there is nothing to diverge.
    calibration = _calibration(
        resolved,
        placeholder.acceptance_thresholds.minimum_calibration_coverage,
        placeholder.acceptance_thresholds.maximum_calibration_error,
    )
    seed = GSE274113ObservationEstimator(
        fold, query=placeholder, calibration=calibration, **digests
    )
    estimator = GSE274113ObservationEstimator(
        fold,
        query=arm_query(slice_data.targets, model_fingerprint=seed.model_fingerprint),
        calibration=calibration,
        **digests,
    )
    return fold, estimator


def available_arms(*, directory: Path | None = None) -> tuple[tuple[str, str], ...]:
    """Every ``(library, target)`` this backend can answer for."""

    slice_data = load_arm_slice(directory or artifact_directory())
    return tuple(
        (library, target) for library in slice_data.libraries for target in slice_data.targets
    )


def estimate_arm(library: str, target: str, *, directory: Path | None = None) -> CellStateBelief:
    """Estimate one arm's state, through the public API, from the fold that never saw its library.

    The fold discipline is not optional and is not the caller's to choose: a belief about an arm in
    library *L* is emitted only by the fold that excluded *L*, and that is wired in here rather than
    left as something a caller could get wrong.
    """

    resolved = directory or artifact_directory()
    slice_data = load_arm_slice(resolved)
    _, estimator = _fold_and_estimator(library, resolved)
    if (library, target) not in slice_data.counts:
        raise ValueError(
            f"unknown arm ({library!r}, {target!r}); "
            f"the slice carries {len(slice_data.targets)} targets per library"
        )
    composition, depth = slice_data.log_composition(library, target)
    request = arm_request(
        library,
        target,
        query=estimator._query,
        log_composition=tuple(float(value) for value in composition),
        cells=slice_data.cells[(library, target)],
        panel_total=int(depth),
    )
    return estimate_cell_state(request, estimator=estimator)


@dataclass(frozen=True)
class GeneLoading:
    """One gene's weight on one biology axis."""

    symbol: str
    loading: float


@dataclass(frozen=True)
class AxisReadout:
    """One biology axis: where this arm sits on it, and which genes define it."""

    name: str
    coordinate: float
    standard_deviation: float
    top_genes: tuple[GeneLoading, ...]

    def __str__(self) -> str:
        genes = ", ".join(f"{gene.symbol}{gene.loading:+.2f}" for gene in self.top_genes)
        return f"{self.name:<11} {self.coordinate:+7.3f} ± {self.standard_deviation:5.3f}   {genes}"


@dataclass(frozen=True)
class StateDescription:
    """A belief's biology block, in gene terms, with its abstention carried through."""

    library: str
    target: str
    axes: tuple[AxisReadout, ...]
    abstention_required: bool
    reasons: tuple[str, ...]

    def __str__(self) -> str:
        header = f"{self.library}/{self.target}  biology state, top-loading genes per axis"
        body = "\n".join(f"  {axis}" for axis in self.axes)
        if not self.abstention_required:
            return f"{header}\n{body}"
        why = "\n".join(f"    - {reason}" for reason in self.reasons)
        return f"{header}\n{body}\n  ABSTENTION REQUIRED:\n{why}"


def describe_state(
    belief: CellStateBelief, *, directory: Path | None = None, top_genes: int = 6
) -> StateDescription:
    """Name each biology axis by the genes that define it, and place this arm on it.

    The axis names a belief carries are ``biology_0`` through ``biology_3``.  The basis is a
    100-gene loading matrix held on the fold, so the genes with the largest absolute weight on each
    column are what that column *is*.  Sorting by absolute value keeps both poles of an axis
    visible, which matters because the informative axes here are contrasts -- a gene at -0.31 is as
    much a part of the axis as one at +0.33.
    """

    if top_genes < 1:
        raise ValueError("top_genes must be at least one")
    prefix, _, remainder = belief.subject.subject_id.partition(":")
    library, _, target = remainder.partition(":")
    if prefix != "gse274113" or not library or not target:
        raise ValueError(
            f"belief subject {belief.subject.subject_id!r} did not come from this backend"
        )
    resolved = directory or artifact_directory()
    fold, _ = _fold_and_estimator(library, resolved)
    symbols = load_arm_slice(resolved).gene_symbols
    basis = fold.biology_basis

    biology = next(
        (factor for factor in belief.factors if factor.factor is StateFactor.REGULATORY), None
    )
    # Narrowed rather than assumed: the factor's posterior is a union, and a belief that abstained
    # into an UnavailableDistribution has no coordinates to report.  Saying so beats an
    # AttributeError four frames down.
    if biology is None or not isinstance(biology.posterior, ParametricDistribution):
        raise ValueError("this belief carries no parametric regulatory posterior to describe")
    posterior = biology.posterior
    if posterior.mean is None:
        raise ValueError("this belief's regulatory posterior carries no mean")
    covariance = posterior.covariance

    axes: list[AxisReadout] = []
    for index, name in enumerate(posterior.dimensions):
        column = basis[:, index]
        order = np.argsort(-np.abs(column))[:top_genes]
        variance = 0.0 if covariance is None else float(covariance[index][index])
        axes.append(
            AxisReadout(
                name=name,
                coordinate=float(posterior.mean[index]),
                standard_deviation=float(np.sqrt(max(variance, 0.0))),
                top_genes=tuple(
                    GeneLoading(symbol=symbols[position], loading=float(column[position]))
                    for position in order
                ),
            )
        )
    return StateDescription(
        library=library,
        target=target,
        axes=tuple(axes),
        abstention_required=belief.readiness.abstention_required,
        reasons=belief.readiness.reasons,
    )


@dataclass(frozen=True)
class ArmContrast:
    """Two arms of the same library, differenced in their shared biology coordinates."""

    library: str
    target_a: str
    target_b: str
    per_axis: tuple[float, ...]
    axis_names: tuple[str, ...]
    distance: float
    distance_lower_bound_sd: float

    def __str__(self) -> str:
        parts = ", ".join(
            f"{name}{value:+.3f}"
            for name, value in zip(self.axis_names, self.per_axis, strict=True)
        )
        return (
            f"{self.library}: {self.target_b} - {self.target_a}  "
            f"|delta| = {self.distance:.3f}  (sd >= {self.distance_lower_bound_sd:.3f})  [{parts}]"
        )


def compare_arms(
    library: str, target_a: str, target_b: str, *, directory: Path | None = None
) -> ArmContrast:
    """Difference two arms of the **same library** in shared biology coordinates.

    Both arms must be in one library, and that is a correctness constraint rather than a
    convenience.  A belief about library *L* is emitted by the fold that excluded *L*, so arms in
    different libraries are expressed in **different fitted bases** and their coordinates are not
    comparable.  The signature makes that impossible to express rather than checking for it.

    ⚠️ ``distance_lower_bound_sd`` is named for what it is.  It adds the two posterior variances as
    though the arms were independent, which they are not: both were scored under the same fold, so
    they share its estimation error, and the true spread of the contrast is **wider** than this.  A
    properly grouped interval for a contrast like this is what ``evaluation/gse274113_reports.py``
    computes by bootstrapping across libraries; this is a per-pair readout, not a substitute for it.
    """

    if target_a == target_b:
        raise ValueError("comparing an arm against itself measures nothing")
    first = describe_state(
        estimate_arm(library, target_a, directory=directory), directory=directory
    )
    second = describe_state(
        estimate_arm(library, target_b, directory=directory), directory=directory
    )
    difference = tuple(
        second_axis.coordinate - first_axis.coordinate
        for first_axis, second_axis in zip(first.axes, second.axes, strict=True)
    )
    variance = sum(
        first_axis.standard_deviation**2 + second_axis.standard_deviation**2
        for first_axis, second_axis in zip(first.axes, second.axes, strict=True)
    )
    return ArmContrast(
        library=library,
        target_a=target_a,
        target_b=target_b,
        per_axis=difference,
        axis_names=tuple(axis.name for axis in first.axes),
        distance=float(np.linalg.norm(difference)),
        distance_lower_bound_sd=float(np.sqrt(variance)),
    )
