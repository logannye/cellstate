# Estimate a real cell state

From a source checkout, with no download and no configuration:

```python
from cellstate.backends.gse274113 import estimate_arm, describe_state, compare_arms

print(describe_state(estimate_arm("rep1", "GATA1")))
print(compare_arms("rep1", "NT", "GATA1"))
```

The whole thing runs in well under a second — loading the committed slice is instantaneous and
fitting a fold takes about ten milliseconds. `python examples/estimate_real_cell_state.py` runs the
same thing end to end.

## What comes back

```
rep1/GATA1  biology state, top-loading genes per axis
  biology_0    +2.011 ± 0.339   MPO+0.33, CD79A-0.31, ELANE+0.28, THBS1-0.26, AZU1+0.21, PPBP-0.21
  biology_1    +0.353 ± 0.367   PRTN3+0.44, ELANE+0.30, PPBP+0.28, PF4+0.26, MPL-0.26, S100A9-0.22
  biology_2    +1.040 ± 0.370   VWF+0.46, CD34-0.35, GP1BA+0.28, S100A8+0.25, CD79A-0.24, IL5RA-0.22
  biology_3    +1.825 ± 0.458   S100A8+0.49, CSF1R+0.32, S100A9+0.27, GP9-0.25, CRHBP+0.25, CTSG+0.22
  ABSTENTION REQUIRED:
    - predictive sufficiency is inapplicable: no library spans the inference cutoff
    - this belief is a snapshot state estimate, not a faithfulness verdict
```

Four biology coordinates, each named by the panel genes that define its direction. `biology_0`
separates primary-granule genes (`MPO`, `ELANE`, `AZU1`) from B-cell and platelet markers (`CD79A`,
`PPBP`); `biology_2` runs megakaryocyte (`VWF`, `GP1BA`) against `CD34`. These axes were fitted from
the data, not declared.

**The readout reports loadings and never a label.** Calling `biology_0` "the granulocyte axis" is a
reasonable interpretation and it is yours to make; the library will not put it in a field, because
an asserted label is a claim no measurement backs.

## Contrasts

```
rep1: GATA1 - NT  |delta| = 1.909  (sd >= 1.035)  [biology_0+0.784, biology_1+0.327, ...]
rep1: TAL1  - NT  |delta| = 1.501  (sd >= 1.029)
rep1: SPI1  - NT  |delta| = 0.765  (sd >= 1.018)
rep1: SNAI2 - NT  |delta| = 0.689  (sd >= 0.998)
```

`compare_arms` takes **one** library and two targets, and that is a correctness constraint rather
than a convenience. A belief about library *L* is emitted by the fold that excluded *L*, so arms in
different libraries are expressed in different fitted bases and their coordinates are not
comparable. The signature makes the incomparable case impossible to write.

## How to read these numbers, and how not to

`SNAI2` is the reference you want. It is measured at **0 CPM** in this panel — the gene is not
expressed, so its contrast is the size of a difference that means nothing here. Read the others
against it rather than against zero.

Three things this surface does **not** establish:

- **`sd >=` is a declared lower bound.** It adds the two posterior variances as though the arms were
  independent. They are not — both were scored under the same fold and share its estimation error —
  so the true spread is wider.
- **A per-pair readout is not a grouped interval.** The measurement with an interval bootstrapped at
  the library lives in `evaluation/gse274113_reports.py`, and on that evidence the perturbed contrast
  is **not** separable from placebo.
- **Every belief here abstains**, and `describe_state` reprints the reasons. A coordinate is not an
  answer to a predictive question.

The honest summary is that the same panel and pipeline resolve the day 7 → day 14 differentiation
contrast at **7.97×** the placebo contrast, while the perturbation contrast moves it far less. What
this repository used to conclude from that — that the deposit's perturbation arm carries almost no
signal — is **withdrawn**: it rested on an on-target-mRNA criterion that belongs to CRISPRi, and
this deposit is Cas9 nuclease knockout, where the transcript need not move. See
[the model card](../backends/gse274113-rna-observation-model.md) for what that does and does not
license.

## Where to go next

- `available_arms()` lists every `(library, target)` the backend can answer for — 14 libraries × 20
  targets.
- `load_arm_slice()` returns the committed slice if you want the counts directly.
- Every entry point takes an optional `directory=` if your artifacts live elsewhere; a wheel does not
  ship the `backends/` tree.
