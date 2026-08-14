# A local web UI for the GSE274113 backend

**Status:** approved 2026-08-14. Implementation follows in the same branch.

## Why

`scripts/explore.py` made the representation inspectable, but it answers one question per
invocation and prints text. Two things it cannot do are exactly the things worth doing next:

- **Vary a model parameter and watch a measurement respond.** `BIOLOGY_RANK` and `NUISANCE_RANK`
  are hardcoded constants. The published rank sensitivity (S5 at 16.96 / 19.22 / 27.21 for ranks
  3 / 4 / 5) had to be produced by editing the source. That is the "iterate on the design" loop, and
  it is currently manual.
- **Show a shape rather than a number.** The two findings that matter most on this substrate are
  both geometric — a singular-value spectrum that does not decay, and a set of fitted bases that
  disagree about which direction `biology_0` points. Both are far clearer drawn than tabulated.

## What it is not

Read-only, single-user, local. No authentication, no persistence, no export, no write path, no
multi-user session state. It serves a slice pinned by SHA-256 and computes with the shipped
estimator; it originates no numbers of its own.

## The one change to the package

`BIOLOGY_RANK` and `NUISANCE_RANK` are read directly inside `fit_fold`
(`backends/gse274113/fit.py:228,238`). They become optional keyword arguments:

```python
def fit_fold(slice_data, held_out_library, *,
             biology_rank: int = BIOLOGY_RANK,
             nuisance_rank: int = NUISANCE_RANK) -> FittedFold
```

Defaults preserve every pinned measurement, and all seven existing call sites are untouched.
`FittedFold` gains no fields — the ranks are already recoverable as `biology_basis.shape[1]` and
`nuisance_basis.shape[1]`.

Two alternatives were rejected. Monkeypatching the module constants per request is not thread-safe
and would leave the server unable to say honestly which ranks produced a result. Reimplementing the
fit inside the server would break the rule that every number comes from the shipped path.

Ranks are validated at the boundary: `biology_rank + nuisance_rank` must be at least 1 and must not
exceed the panel width, and the fit must leave the target directions non-degenerate.

## Architecture

```
make ui  ->  uvicorn  ->  FastAPI  (src/cellstate/ui/server.py)
                            |  read-only JSON over HTTP, localhost only
                            v
                          cellstate.backends.gse274113        fit_fold, estimate_arm, describe_state
                          cellstate.evaluation.gse274113_reports   S2 / S4 / S5
                            ^
                          lru_cache keyed on (biology_rank, nuisance_rank)
```

**FastAPI, not Flask.** `pydantic` is already a core dependency, so response bodies are declared as
typed models in the same contract-first style as the rest of the repository rather than
hand-assembled dicts. It lives in a `ui` optional extra; the core install stays numpy / pydantic /
scipy.

**Vanilla frontend, no build step, no CDN.** One page served static by the same process. Adding a
node toolchain to a Python research repository is a large permanent footprint for a local tool, and
a CDN dependency would make the UI fail offline. Files stay small and single-purpose:
`static/index.html`, `static/app.js`, `static/charts.js`, `static/style.css`.

**Cost.** One fold is ~10 ms, fourteen ~140 ms, a full `measure` ~1 s. Cached per rank pair, so a
slider drag is instant after the first touch of each value.

## Endpoints

| Route | Returns |
|---|---|
| `GET /api/inventory` | libraries with day and cell counts, targets, panel size, rank defaults |
| `GET /api/panel` | 100 genes with mean NT expression and target/not-expressed flags |
| `GET /api/arm/{library}/{target}` | belief: per-axis coordinate, sd, top loadings, abstention reasons |
| `GET /api/library/{library}/states` | every arm's biology coordinates + covariance, for the scatter |
| `GET /api/fold/{library}` | W and V loadings, ψ² and its pre-clamp value, per-target &#124;u_g&#124; |
| `GET /api/sweep/{library}` | every target against NT, with the not-expressed floor |
| `GET /api/ranks` | cross-library rank table and the per-library winner |
| `GET /api/measure` | S2, S4 both halves, S5, and the three-term decomposition |
| `GET /api/rank-response` | S5 and both component terms across the whole rank grid |
| `GET /api/spectrum` | perturbation / placebo / differentiation singular values |
| `GET /api/knockdown` | per-target on-target log2FC and NT expression |
| `GET /api/day` | differentiation readout and per-gene day correlation |
| `GET /api/basis` | cross-fold cosine per axis, anchor genes, principal angles |

Every route that depends on the fit accepts `biology_rank` and `nuisance_rank` query parameters.

## Views

**Model laboratory.** Rank sliders driving a live refit. Beside the current S5 and decomposition, a
response curve of S5 and both component terms across the rank grid with the current position marked,
and the singular-value spectrum with the cut drawn where the rank falls. The point is that the cut
lands in a flat region.

**Arm explorer.** Library × target picker. Each biology axis drawn as a diverging bar of its
top-loading genes, which is what an axis is. Below, the arms of that library plotted in a selectable
coordinate pair with posterior uncertainty ellipses. That plot is expected to show the arms
overlapping, which is the substrate finding rather than a rendering failure.

**Substrate screens.** Knockdown as a sorted diverging bar with wrong-signed targets distinguished;
the spectrum decay chart; the day readout; the cross-library rank table.

**Fold / basis inspector.** A 14x14 cross-fold cosine heatmap per axis on a diverging scale, where
the sign flips read as a checkerboard; the anchor-gene table showing `biology_0` alternating
MPO ↔ CD79A; principal angles between fold subspaces.

## Honesty constraints the UI inherits

These are not stylistic. They are the same lines `usage.py` and `explore.py` hold.

- **Loadings, never labels.** No `axis_label` field, no "granulocyte axis" caption.
- **Abstention is reprinted, never suppressed.** Every belief this backend emits abstains; the
  reasons appear with the coordinates, not behind a disclosure.
- **No coordinate is compared across libraries.** The contrast control accepts one library and two
  targets. The cross-fold heatmap is the one place fold bases meet, and its entire purpose is to
  show that they disagree.
- **A lower bound is labelled as one.** `distance_lower_bound_sd` is drawn and named as a lower
  bound, never as an interval.
- **A non-default rank is marked.** Any panel showing a measurement computed at other than
  `BIOLOGY_RANK=4` / `NUISANCE_RANK=3` says so, so a screenshot cannot be mistaken for the
  published number.

## Testing

`tests/test_ui_server.py`, driven by `fastapi.testclient.TestClient`:

- every endpoint returns 200 with a body matching its response model;
- default ranks reproduce the pinned S5 and decomposition *through the API*, so the HTTP layer is
  shown not to alter them;
- a non-default rank returns a **different** S5 — the slider is proven not to be inert, which is the
  `do`-operator lesson applied to a UI control;
- an unknown library, an unknown target, and a degenerate rank are each refused with 4xx;
- `fit_fold`'s new parameters are exercised directly in the backend test file, asserting the
  defaults still reproduce the pinned numbers.

## Out of scope

Static export or sharing, editing the slice, uploading data, comparing two rank settings
side-by-side, and any second backend. Each is a separate request.
