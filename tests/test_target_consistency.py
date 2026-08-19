"""The within-library target-label permutation screen.

This is the first positive evidence on `GSE274113`'s perturbation arm since the "measured null"
verdict was withdrawn (PR #41, PR #42). It is deliberately model-free: no fold, no fitted basis, no
observation-variance model, no bound. It asks one question -- do arms sharing a target agree with
each other more than the same arms agree under a relabelling that holds the library fixed?

It carries **no verdict**, and `PermutationScreen` has no `passed` field. That is a design choice,
not an omission: this repository has one criterion that has ever been observed passing, and a screen
that emitted a verdict against an unwitnessed threshold would be the defect it exists to help
diagnose. See ADR 0026 (proposed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cellstate.backends.gse274113.fit import ArmSlice
from cellstate.evaluation.target_consistency import (
    PermutationScreen,
    screen_target_consistency,
)

ROOT = Path(__file__).resolve().parents[1]
SLICE_PATH = ROOT / "backends" / "vertical-a" / "gse274113-rna-obs-v1" / "arms.json"


@pytest.fixture(scope="module")
def arm_slice() -> ArmSlice:
    return ArmSlice.from_payload(json.loads(SLICE_PATH.read_text(encoding="utf-8")))


def test_the_permutation_null_lands_where_theory_says_it_must(arm_slice: ArmSlice) -> None:
    """A positive control on the SCREEN, not on the data.

    Under the null the target labels are exchangeable within a library, so the target main effect
    captures the share of within-library variance that any arbitrary K-way grouping of K libraries
    would capture by chance -- which is approximately ``1 / K``. With fourteen libraries that is
    0.0714.

    If the null does not land there the construction is wrong, and the observed value it is
    compared against means nothing. This assertion is what makes the screen's own arithmetic
    falsifiable rather than merely plausible.
    """

    screen = screen_target_consistency(arm_slice, draws=400, seed=1)
    expected = 1.0 / len(arm_slice.libraries)
    assert screen.null_mean == pytest.approx(expected, rel=0.05), (
        f"permutation null {screen.null_mean:.4f} should sit at 1/K = {expected:.4f}; "
        "a null elsewhere means the statistic is not measuring what it claims"
    )


def test_targets_agree_with_each_other_more_than_relabelling_explains(arm_slice: ArmSlice) -> None:
    """The measurement. Pinned loosely because it is a Monte Carlo estimate, not a fixture."""

    screen = screen_target_consistency(arm_slice, draws=2000, seed=20260819)

    assert screen.observed == pytest.approx(0.1897, abs=0.002)
    assert screen.ratio_to_null > 2.0
    assert screen.exceedances == 0
    # (0 + 1) / (2000 + 1): the floor of what this many draws can resolve, never a bare zero.
    assert screen.p_value == pytest.approx(1.0 / 2001.0, rel=1e-6)


def test_the_screen_is_reproducible_from_its_declared_seed(arm_slice: ArmSlice) -> None:
    first = screen_target_consistency(arm_slice, draws=200, seed=5)
    second = screen_target_consistency(arm_slice, draws=200, seed=5)
    assert first == second

    different = screen_target_consistency(arm_slice, draws=200, seed=6)
    assert different.observed == first.observed, "the observed value must not depend on the seed"
    assert different.null_mean != first.null_mean, "the null is resampled, so it must move"


def test_a_slice_with_no_target_structure_lands_inside_its_own_null(
    arm_slice: ArmSlice,
) -> None:
    """The discriminating control: the screen must be able to return nothing.

    A screen that reports structure on every input is not evidence of structure. Here the target
    labels are shuffled within each library BEFORE the screen runs, which destroys target identity
    while leaving every count vector, every library effect and every depth exactly as deposited.
    """

    rng = np.random.default_rng(11)
    perturbed = [t for t in arm_slice.targets if t != "NT"]
    counts = dict(arm_slice.counts)
    for library in arm_slice.libraries:
        shuffled = list(rng.permutation(perturbed))
        originals = {t: arm_slice.counts[(library, t)] for t in perturbed}
        for target, source in zip(perturbed, shuffled, strict=True):
            counts[(library, target)] = originals[source]
    scrambled = ArmSlice(
        gene_symbols=arm_slice.gene_symbols,
        libraries=arm_slice.libraries,
        targets=arm_slice.targets,
        library_day=arm_slice.library_day,
        counts=counts,
        cells=arm_slice.cells,
    )

    screen = screen_target_consistency(scrambled, draws=400, seed=3)
    assert screen.observed < screen.null_upper, (
        f"a label-scrambled slice returned {screen.observed:.4f} against a null upper of "
        f"{screen.null_upper:.4f}; the screen reports structure that is not there"
    )
    assert screen.p_value > 0.01


def test_the_screen_carries_no_verdict(arm_slice: ArmSlice) -> None:
    """ADR 0026's shape, applied to this screen before the ADR exists.

    A measurement without a witnessed threshold must not emit a pass. The absence of a `passed`
    field is the enforcement, so that a future caller cannot read one off by accident.
    """

    screen = screen_target_consistency(arm_slice, draws=100, seed=2)
    assert not hasattr(screen, "passed")
    assert not hasattr(screen, "outcome")
    assert isinstance(screen, PermutationScreen)
