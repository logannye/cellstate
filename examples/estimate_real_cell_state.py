"""Estimate the state of real human CD34+ cells, and read what the estimate says.

Run it with ``python examples/estimate_real_cell_state.py`` from a source checkout.  It needs no
download and no configuration: the panel and the arm slice are committed, and the whole thing
finishes in well under a second.

The sibling example, ``estimate_state.py``, drives the *non-biological reference* backend and takes
366 lines to construct its inputs by hand.  This one is short because the work happens in
:mod:`cellstate.backends.gse274113.usage`, not because it is doing less.

⚠️ Read the abstention.  Every belief here refuses to be treated as a prediction, and the reasons
are printed rather than suppressed.  The contrasts below are per-pair readouts whose reported spread
is a *lower bound* -- the properly grouped intervals live in ``evaluation/gse274113_reports.py``,
and on that evidence the perturbed contrast is **not** separable from placebo.  What this example
demonstrates is that the representation can be inspected, not that the perturbation is resolved.
"""

from __future__ import annotations

from cellstate.backends.gse274113 import (
    available_arms,
    compare_arms,
    describe_state,
    estimate_arm,
)

LIBRARY = "rep1"


def main() -> None:
    arms = available_arms()
    libraries = sorted({library for library, _ in arms})
    print(f"{len(arms)} arms across {len(libraries)} libraries; using {LIBRARY}\n")

    # One call. The fold that answers an arm is the fold that never saw its library, and that is
    # wired in rather than left to the caller.
    belief = estimate_arm(LIBRARY, "GATA1")
    print(describe_state(belief))

    # Each biology axis is a direction in the 100-gene panel, so the genes with the largest weight
    # on it are what it is. Nothing here labels an axis -- the loadings are the readout.
    print("\ncontrasts against the non-targeting control, same library:\n")
    for target in ("GATA1", "TAL1", "SPI1", "SNAI2"):
        print(f"  {compare_arms(LIBRARY, 'NT', target)}")

    print(
        "\nSNAI2 is the useful comparison: it is not expressed in this panel (0 CPM), so its\n"
        "contrast is the size of a difference that means nothing here. Read the others against it\n"
        "rather than against zero."
    )


if __name__ == "__main__":
    main()
