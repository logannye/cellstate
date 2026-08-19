"""What fraction of the differentiation axis does each fold's nuisance block absorb?

`fit.py` states its identification argument as a fact: *"NT is the same biology in every library,
so whatever moves there is the library."* It is not. The fourteen libraries sit at four different
differentiation days (7, 9, 11, 14), so `NT` at day 7 and `NT` at day 14 are not the same biology,
and the across-library `NT` residual that `V` is fitted on contains the culture's differentiation
clock.

The premise is not repaired here -- repairing it is a change to what the biology block means and
needs its own decision record. What is repaired is that the premise was *asserted* rather than
measured. Every fold now reports the number, so a reader can see what the block absorbs instead of
taking a docstring's word for it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cellstate.backends.gse274113.fit import ArmSlice, fit_fold

ROOT = Path(__file__).resolve().parents[1]
SLICE_PATH = ROOT / "backends" / "vertical-a" / "gse274113-rna-obs-v1" / "arms.json"


@pytest.fixture(scope="module")
def arm_slice() -> ArmSlice:
    return ArmSlice.from_payload(json.loads(SLICE_PATH.read_text(encoding="utf-8")))


def test_every_fold_reports_how_much_of_the_day_axis_its_nuisance_block_absorbs(
    arm_slice: ArmSlice,
) -> None:
    """Measured, not asserted: 0.999 in fourteen of fourteen folds.

    A random three-dimensional subspace of a hundred-gene space would capture about 0.03. This is
    not a subspace that happens to contain some differentiation signal; it is the differentiation
    axis, with the library-within-day component alongside it.
    """

    for library in arm_slice.libraries:
        fold = fit_fold(arm_slice, library)
        assert fold.day_axis_in_nuisance_basis == pytest.approx(0.999, abs=0.005), (
            f"fold {library} absorbs {fold.day_axis_in_nuisance_basis:.4f} of the day axis"
        )


def test_the_biology_block_is_left_with_almost_none_of_it(arm_slice: ArmSlice) -> None:
    """The other half of the same fact, and the reason it matters.

    `W` is orthogonalized against `V` (``fit.py``), so whatever `V` absorbs is removed from biology
    by construction. With `V` holding 0.999 of the day axis, the biology block is left with
    essentially none of the strongest biological signal in the deposit.
    """

    fold = fit_fold(arm_slice, arm_slice.libraries[0])
    assert fold.day_axis_in_biology_basis < 0.01
    assert fold.day_axis_in_nuisance_basis > 100 * fold.day_axis_in_biology_basis


def test_sampling_noise_does_not_project_into_the_nuisance_block(arm_slice: ArmSlice) -> None:
    """The discriminating control: `V` is not simply a large subspace that absorbs everything.

    The placebo contrast ``NT_B - NT_A`` is a split of the same cells, so it is sampling noise by
    construction. If `V` absorbed it at the same rate it absorbs the day axis, the 0.999 would say
    only that three dimensions catch a lot, and would carry no information about what they catch.
    """

    fold = fit_fold(arm_slice, arm_slice.libraries[0])
    placebo = []
    for library in fold.fit_library_ids:
        if all((library, half) in arm_slice.counts for half in ("NT_A", "NT_B")):
            first, _ = arm_slice.log_composition(library, "NT_A")
            second, _ = arm_slice.log_composition(library, "NT_B")
            direction = second - first
            direction = direction / np.linalg.norm(direction)
            placebo.append(float(np.sum((fold.nuisance_basis.T @ direction) ** 2)))

    assert placebo, "the placebo halves are needed for this control"
    assert float(np.mean(placebo)) < 0.15, (
        f"noise projects {np.mean(placebo):.3f} into V, against 0.999 for the day axis; "
        "if these were comparable the day figure would carry no information"
    )


def test_a_fold_from_a_single_day_reports_an_undefined_day_axis(arm_slice: ArmSlice) -> None:
    """The diagnostic must refuse rather than invent a number when there is no day contrast.

    A slice whose fit libraries all sit at one day has no day-7-to-day-14 direction to project, and
    a fold reporting 0.0 there would be indistinguishable from one that genuinely absorbs nothing.
    """

    day_seven = tuple(lib for lib in arm_slice.libraries if arm_slice.library_day[lib] == 7)
    assert len(day_seven) >= 3, "this control needs several libraries at one day"

    single_day = ArmSlice(
        gene_symbols=arm_slice.gene_symbols,
        libraries=day_seven,
        targets=arm_slice.targets,
        library_day={lib: arm_slice.library_day[lib] for lib in day_seven},
        counts={k: v for k, v in arm_slice.counts.items() if k[0] in day_seven},
        cells={k: v for k, v in arm_slice.cells.items() if k[0] in day_seven},
    )
    fold = fit_fold(single_day, day_seven[0])
    assert fold.day_axis_in_nuisance_basis is None
    assert fold.day_axis_in_biology_basis is None
