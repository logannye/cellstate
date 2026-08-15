# The web UI

```bash
make ui        # syncs the extra, then serves http://127.0.0.1:8000
```

A local, read-only browser surface over the committed GSE274113 slice. It computes with the shipped
estimator and originates no numbers of its own; at the fitted configuration every value it shows is
the value in the model card.

It exists for the one thing [`scripts/explore.py`](explore-the-system.md) cannot do: **vary a model
parameter and watch a measurement respond.** `BIOLOGY_RANK` and `NUISANCE_RANK` used to be
constants, so producing the published rank sensitivity meant editing `fit.py` — not something a
reader can do while looking at the number it moves.

Read-only, single user, bound to `127.0.0.1`. No authentication, no persistence, no write path.

## Model laboratory

Two sliders, a live refit, and the measurements underneath.

The **response curve** plots S5's two component terms across the whole rank grid with your current
position marked. The published sensitivity is three points of this curve. What the whole curve shows
is why the ratio is a poor summary: the leakage term falls steadily as rank rises and the
between-target signal falls *with* it, so S5 barely moves.

The **spectrum panel** draws the singular values of the matrix `W` is fitted on, with a dashed line
where your rank cuts. Drag the slider and watch the line sweep a region with no gap in it.

> ⚠️ **Away from `4 / 3` the page says so, in a banner and on every affected panel.** S2 and S4
> disappear entirely at other ranks — their estimands are fixed by merged ADRs at the fitted
> configuration, and recomputing them elsewhere would present a number no decision authorises. S5
> is shown as a point estimate with **no interval**, because the bound is predeclared for the
> fitted configuration and this is not it.

## Arm explorer

Pick any library and target. Each biology axis is drawn as a **diverging bar of its top-loading
genes** — both poles, because the informative axes here are contrasts. Loadings, never labels: the
readout will tell you `biology_0` loads `MPO +0.33` and `CD79A −0.31`, and will not call that "the
granulocyte axis".

Beside it, every arm of that library plotted in two of its shared coordinates with **1-sd posterior
ellipses**. Expect them to overlap heavily. That is the substrate finding, not a rendering problem.

The abstention travels with the coordinates rather than sitting behind a disclosure. Every belief
this backend emits abstains — and since ADR 0024 it says *why*: the reasons now carry S6's measured
coverage and bound, and an explicit list of the six criteria still unevaluated.

> One library at a time, by construction. A belief about library *L* comes from the fold that
> excluded *L*, so arms in different libraries are expressed in different fitted bases and their
> coordinates do not compare.

## Substrate

**S6 leads the view**, because it is the only readiness criterion this backend evaluates: the
coverage headline with its bound against the predeclared 0.05, **the table of all six gated nominal
levels**, the trimmed-tail bars against a reference line at 1.00, and per-library coverage plotted
against panel depth with the nominal drawn across. It takes no rank arguments and the panel says so
— S6's estimand is fixed at the fitted configuration, so a knob there would imply a measurement that
does not exist.

The verdict shown is the **conjunction over the six levels**, never the headline row alone; the page
cannot display a pass or fail without displaying what produced it. Per
[ADR 0025](../adr/0025-s6-is-gated-on-the-whole-coherent-nominal-interval.md), gating at one level
would have been clearable by a constant.

Below it, the three independent statements that this deposit carries no perturbation signal —
`knockdown` as a sorted diverging bar, the `spectrum` decay chart, the `day` readout — plus the
cross-library rank table and the panel's expression profile.

## Fold / basis

The declared limit, drawn. A **14 × 14 cosine heatmap** of how much each pair of fitted bases agrees
about which direction the selected axis points. Teal is agreement, rust is anti-alignment. For
`biology_0` it reads as a checkerboard: 48 of 91 fold pairs disagree about the sign, because
`_canonical_signs` keys the sign to the largest-magnitude entry and that entry's *identity*
alternates between `MPO` and `CD79A` — opposite poles of the same axis, differing by under 1%.

Beside it, the anchor gene per fold, and below, that fold's fitted `W` and `V`, its ψ², and each
target's residual norm.

This is the limit S5 and the block decomposition are computed under: both pool coefficients across
all fourteen of these bases. Recorded in the model card, not repaired — correcting it moves S5 from
10.36 to 9.23 and the between-target signal from 0.109 to 0.129, which is a change to a published
measurement and belongs in an ADR.

## The API

Every view is backed by a documented JSON endpoint; `http://127.0.0.1:8000/api/docs` is the generated
reference. Each route that depends on the fit accepts `biology_rank` and `nuisance_rank`, and every
response carries the ranks it was computed at.

```bash
curl -s localhost:8000/api/measure | jq '.measurements[].value'
curl -s 'localhost:8000/api/measure?biology_rank=6' | jq '.ranks'
```

## What it will refuse

- an unknown library or target — `404`
- a rank below 1, or a pair whose sum reaches the panel width — `422`
- **a rank larger than the fold can resolve** — `422`. A nuisance rank above 13 would previously
  have returned a *smaller* basis than requested without saying so, since `svd` yields only
  `min(rows, columns)` vectors. That silent truncation is now refused; it was found by building
  this UI.
