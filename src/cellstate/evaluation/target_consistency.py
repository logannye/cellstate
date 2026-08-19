"""Does the perturbation arm carry target-consistent structure?  A model-free permutation screen.

Every other capability measurement in this package runs through the fitted observation model: a
leave-one-library-out fold, a biology basis, an observation-variance model, a predeclared bound.
That is the right way to measure a *capability*, and it is the wrong way to answer a prior question
this repository has been unable to settle -- **whether the deposit carries any target-consistent
signal at all**, independently of the estimator that has so far failed to find it.

The question matters because the two available explanations for a failing ledger are not
distinguishable from the ledger itself.  A criterion can fail because the substrate is empty, or
because the estimator cannot see what is there.  For weeks this project believed the first, on the
strength of an on-target-mRNA control calibrated to the wrong perturbation technology; that verdict
is withdrawn (`docs/data/representability/gse274113-perturb-multiome.md`), and withdrawing it left
the question genuinely open rather than answered the other way.

**The construction.**  For each library *L* and perturbed target *g*, take the within-library
contrast ``c[L, g] - c[L, NT]`` in log-composition space.  Differencing inside a library removes the
library main effect exactly rather than modelling it.  Centre those contrasts within each library,
then ask what share of the total sum of squares is carried by the *target means* -- the agreement
between arms that share a target, across libraries.

Compare that share against the distribution it takes when the target labels are permuted **within
each library**.  Permuting inside the library holds the library structure, the depths and the count
vectors exactly fixed, so the null differs from the observation in one respect only: which arm is
called which target.

**What it does and does not license.**  It is evidence that arms sharing a target agree more than
relabelling explains.  It is not a capability, not an effect size in any interpretable unit, and not
transportable: every arm here comes from one donor's culture (ADR 0018 finding 4), so this is
instrument-and-substrate evidence and never a biological claim.

**It carries no verdict, deliberately.**  ``PermutationScreen`` has no ``passed`` field and no
threshold.  Nine of this repository's ten ledger criteria have never been observed passing on any
substrate, and a screen that emitted a pass against an unwitnessed threshold would reproduce the
defect it exists to help diagnose.  A reader may draw a conclusion; the type will not draw one for
them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..backends.gse274113.fit import NULL_TARGET, PLACEBO_TARGETS, ArmSlice

FloatArray = NDArray[np.float64]

DEFAULT_DRAWS = 2000
DEFAULT_SEED = 20260819

__all__ = [
    "DEFAULT_DRAWS",
    "DEFAULT_SEED",
    "PermutationScreen",
    "screen_target_consistency",
]


@dataclass(frozen=True)
class PermutationScreen:
    """An observed statistic beside the null it was compared against.

    There is no ``passed`` field.  See the module docstring: a measurement whose threshold has never
    been witnessed must not emit a verdict, and the absence of the field is what enforces that
    against a future caller reading one off by accident.
    """

    statistic: str
    observed: float
    null_mean: float
    null_lower: float
    null_upper: float
    draws: int
    exceedances: int
    seed: int
    unit_count: int

    @property
    def p_value(self) -> float:
        """The permutation p-value, floored at ``1 / (draws + 1)``.

        Add-one in both numerator and denominator, so a screen that no draw exceeded reports the
        resolution its draw count actually bought rather than a bare zero.  A p of exactly zero
        would claim more than any finite permutation test can.
        """

        return (self.exceedances + 1) / (self.draws + 1)

    @property
    def ratio_to_null(self) -> float:
        """Observed share over the null's mean share.  Unitless, and the null's mean is ~1/K."""

        return self.observed / self.null_mean if self.null_mean > 0 else float("nan")


def _within_library_contrasts(slice_data: ArmSlice) -> tuple[FloatArray, tuple[str, ...]]:
    """Return ``(libraries x targets x genes)`` within-library contrasts, centred per library.

    ``NT`` is the reference and the placebo halves are excluded: ``NT_A``/``NT_B`` are a
    deterministic split of the same cells ``NT`` already contributes, so including them would let an
    arm agree with itself and inflate the very statistic being screened.
    """

    perturbed = tuple(
        target
        for target in slice_data.targets
        if target != NULL_TARGET and target not in PLACEBO_TARGETS
    )
    if len(perturbed) < 2:
        raise ValueError("target consistency needs at least two perturbed targets to compare")

    rows: list[list[FloatArray]] = []
    for library in slice_data.libraries:
        reference, _ = slice_data.log_composition(library, NULL_TARGET)
        library_rows = []
        for target in perturbed:
            composition, _ = slice_data.log_composition(library, target)
            library_rows.append(np.asarray(composition - reference, dtype=np.float64))
        rows.append(library_rows)

    contrasts = np.asarray(rows, dtype=np.float64)
    # Centre within library, so a per-library offset surviving the NT reference cannot masquerade
    # as target agreement.
    return contrasts - contrasts.mean(axis=1, keepdims=True), perturbed


def _target_share(contrasts: FloatArray) -> float:
    """Share of within-library sum of squares carried by the target means.

    Under within-library label exchangeability the expectation of this statistic is approximately
    ``1 / K`` for ``K`` libraries, which is what ``test_the_permutation_null_lands_where_theory_says
    _it_must`` checks: a null anywhere else means this arithmetic is not measuring what it claims.
    """

    total = float((contrasts**2).sum())
    if total <= 0.0:
        raise ValueError("the contrast matrix is identically zero; there is nothing to screen")
    target_means = contrasts.mean(axis=0)
    return float(contrasts.shape[0] * (target_means**2).sum() / total)


def screen_target_consistency(
    slice_data: ArmSlice,
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
) -> PermutationScreen:
    """Screen the perturbation arm for target-consistent structure, holding the library fixed.

    ``draws`` bounds the resolution: a screen no draw exceeds reports ``p = 1 / (draws + 1)``, so
    2000 draws resolves to 5.0e-4 and no finer.  ``seed`` is recorded on the result, because a
    permutation null that cannot be reproduced is not a null anybody can check.
    """

    if draws < 1:
        raise ValueError("a permutation screen needs at least one draw")

    contrasts, perturbed = _within_library_contrasts(slice_data)
    observed = _target_share(contrasts)

    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        # Permute the target labels INDEPENDENTLY within each library. A single shared permutation
        # would relabel every library the same way, which preserves target agreement exactly and
        # would produce a null indistinguishable from the observation.
        permuted = np.stack(
            [row[rng.permutation(len(perturbed))] for row in contrasts],
            axis=0,
        )
        null[index] = _target_share(permuted)

    lower, upper = (float(value) for value in np.percentile(null, [2.5, 97.5]))
    return PermutationScreen(
        statistic="target share of within-library sum of squares",
        observed=observed,
        null_mean=float(null.mean()),
        null_lower=lower,
        null_upper=upper,
        draws=draws,
        exceedances=int((null >= observed).sum()),
        seed=seed,
        unit_count=len(slice_data.libraries),
    )
