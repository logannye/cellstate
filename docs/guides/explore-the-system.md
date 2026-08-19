# Explore the system

`scripts/explore.py` is a command-line surface over the committed GSE274113 slice. It adds no
capability and makes no claim the package does not already make — every number it prints comes from
the shipped path, or is a plain observational statistic whose one-line construction is written out
in the command's own `--help`.

It exists because a representation nobody can poke at cannot be iterated on, and until now poking at
it meant writing a script each time.

## Launch

From a source checkout, Python 3.11+ with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras
uv run python scripts/explore.py inventory
```

No download and no configuration. The panel and the arm slice are committed; the whole tour below
runs in about three seconds of wall clock, and the slowest single command (`measure`, which fits all
fourteen leave-one-library-out folds and bootstraps four capability measurements) takes about one
second.

`scripts/explore.py` puts `src/` on `sys.path` itself, so it also runs under a bare interpreter that
has `numpy`, `scipy` and `pydantic`:

```bash
PYTHONPATH=src python scripts/explore.py inventory
```

## The tour, in the order worth reading it

### 1. `inventory` — what is actually in the box

```bash
uv run python scripts/explore.py inventory
```

Fourteen libraries, twenty targets, 137,604 cells, a 100-gene panel, and the day each library was
harvested. `NT` is the non-targeting control; `NT_A` / `NT_B` are a deterministic within-library
split of the `NT` cells whose contrast is the placebo floor every other contrast is read against.

### 2. `knockdown` — how much on-target transcript survives the edit

```bash
uv run python scripts/explore.py knockdown
```

For each knockout target that is itself a panel gene, it prints the mean over libraries of
`log2( CPM_target(arm target) / CPM_target(arm NT) )`.

⚠️ **This screen does not say whether the perturbation worked, and it never did.** It gives
**−0.058** with **6 of 19 targets moving the wrong way**, and two targets (`SNAI2` at 0.6 CPM,
`PRDM16` at 3.5 CPM) are not expressed in the panel at all. That was read as a *measured null*
against an expectation of −1 to −2 — which is a CRISPRi number. GSE274113 is **Cas9 nuclease
knockout**: cutting destroys the protein and leaves the transcript largely intact, so a fully
working screen is entirely consistent with these figures. The verdict is **withdrawn**; the
measurement stands, as a measurement of nonsense-mediated-decay escape.

The controls that would settle it — guide-level replication, expression-dependence of effect size,
and the cutting-versus-non-cutting contrast — are not yet implemented. Until they are, treat the
between-target biology variance of 0.109 that `measure` divides by as unexplained rather than as
explained by a dead substrate.

### 2b. `consistency` — do targets agree more than relabelling explains?

```bash
uv run python scripts/explore.py consistency
```

The prior question the ledger cannot answer about itself. A criterion can fail because the substrate
is empty or because the estimator cannot see what is there, and `measure` alone does not distinguish
them.

Model-free: the library effect is removed by differencing each arm against that library's own `NT`,
and the null permutes target labels **within** each library — so the null and the observation differ
in exactly one respect, which arm is called which target.

| quantity | value |
|---|---:|
| observed target share of within-library SS | **0.1897** |
| permutation null (2000 draws) | 0.0714 [0.0615, 0.0821] |
| ratio to null | **2.66×** |
| draws at or above observed | **0 of 2000** |
| p | 5.0e-4 |

The null sits at 1/K for K = 14 libraries, exactly where within-library label exchangeability puts
it. That is a positive control on the *screen*, not the data, and
`test_the_permutation_null_lands_where_theory_says_it_must` asserts it: a null anywhere else means
the statistic is not measuring what it claims.

> ⚠️ **Reports no verdict, on purpose.** `PermutationScreen` has no `passed` field. No threshold on
> this statistic has ever been witnessed, and nine of the ten ledger criteria have never been
> observed passing on any substrate. A screen that emitted a pass against an unwitnessed threshold
> would reproduce the defect it exists to help diagnose.

It establishes that the deposit is **not empty**. It does not establish that the estimator is at
fault, and it is not transportable — every arm comes from one donor's culture (ADR 0018 finding 4).

### 3. `day` — does the panel see biology that is actually there?

```bash
uv run python scripts/explore.py day
```

The control for the control. Same panel, same pipeline, no fitted basis and no regrouping — just
the raw log-composition of the `NT` arms, day 7 against day 14:

| contrast | ‖Δ‖ | vs placebo |
|---|---:|---:|
| `NT` day 7 → day 14 | 40.207 | **7.97×** |
| perturbed target vs `NT` (mean) | 5.383 | 1.07× |
| placebo `NT_A` vs `NT_B` (mean) | 5.046 | 1.00× |

92 of 100 panel genes track day at \|r\| > 0.7. **The instrument works.** Differentiation moves this
panel eight times the placebo contrast; the knockout perturbation moves it 1.07× — which *is* the
placebo floor.

> ⚠️ This is a **readout**, deliberately not a measurement. `library_day` is nested inside `library`
> (three or four libraries per day, none spanning two), so it carries no interval and clears no
> gate. Re-pointing the biology block at this axis would collapse K from 14 to 4 and yield a
> *passing* S5 that means nothing.

### 4. `spectrum` — is there structure for the model to find?

```bash
uv run python scripts/explore.py spectrum
```

`knockdown` asks whether the perturbation moved its own target gene. This asks the harder question:
does the matrix the biology basis `W` is *fitted on* — the within-library contrasts
`c[L, g] − c[L, NT]` — carry any structure at all?

| contrast matrix | rows | s0 | s1 | s2 | s3 | s1/s0 | PC1 var |
|---|---:|---:|---:|---:|---:|---:|---:|
| perturbation: `target − NT` | 266 | 41.0 | 31.1 | 26.2 | 24.8 | 0.76 | 20.1% |
| placebo: `NT_B − NT_A` (noise) | 14 | 10.0 | 7.5 | 6.4 | 5.9 | **0.75** | **26.6%** |
| differentiation: `NT` across days | 14 | 55.4 | 11.1 | 5.9 | 4.9 | **0.20** | **92.5%** |

Real biology *concentrates*: differentiation puts 92% of its variance on one direction. The
perturbation contrast — the matrix `W` is actually fitted on — has the spectral shape of the
**placebo**. ⚠️ This was read as the same verdict `knockdown` reached, independently. It is
not independent corroboration of anything: the placebo contrast is not a noise reference, and
`s1/s0` is not invariant to row count, so the comparison as drawn cannot support a verdict.
That defect is separate from the modality correction and is not yet repaired.

Two consequences for the design:

- **`BIOLOGY_RANK = 4` cuts a flat spectrum.** There is no gap at four to cut at — the first
  *excluded* direction is ~88% the size of the last *included* one. The rank is a free parameter,
  and the published rank sensitivity (S5 at 16.96 / 19.22 / 27.21 for ranks 3 / 4 / 5) is what a
  free parameter cutting noise looks like.
- **Only the leading axis is identified.** `biology_2` and `biology_3` sit on near-degenerate
  singular values, so which direction receives which name is close to arbitrary — and it changes
  between folds. Run `axes rep1` and `axes rep9` and compare: `biology_0`'s top-loading gene is
  `MPO +0.335` in one and `CD79A +0.331` in the other, which are opposite poles of the same axis.

### 5. `state` — one arm's belief, in gene terms

```bash
uv run python scripts/explore.py state rep1 GATA1
```

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

Four biology coordinates, each named by the panel genes that define its direction. The axes were
fitted from the data. **The readout reports loadings and never a label** — calling `biology_0` "the
granulocyte axis" is a reasonable interpretation and it is yours to make; the library will not put
it in a field, because an asserted label is a claim no measurement backs.

The abstention is reprinted, not smoothed over. Every belief this backend emits abstains.

### 6. `axes` — what those coordinates are coordinates *in*

```bash
uv run python scripts/explore.py axes rep1
```

The fitted `W` (biology) and `V` (nuisance) bases of one fold, the fitted `ψ²` and whether it was
clamped, and each target's residual direction norm `|u_g|` — how much that target does *beyond* the
shared basis.

Two things to notice. `W` is orthogonalized against `V`, which is a **declared bias**: biology
genuinely aligned with the library axis is assigned to nuisance, biasing against finding biology
rather than for it. And `NT`'s own `|u_g|` comes out second-largest of all twenty targets — the
placebo split's residual direction is as big as a real perturbation's, which is the null showing up
again in a different statistic.

### 7. `contrast` and `sweep` — differences between arms

```bash
uv run python scripts/explore.py contrast rep1 NT GATA1
uv run python scripts/explore.py sweep rep1
```

`sweep` ranks every target against `NT` in one library, next to the **floor** — the mean contrast of
the targets that are not expressed and therefore cannot have been knocked down. Read every row
against the floor, never against zero. In `rep1` only `GATA1` reaches 1.9× the floor, and the
not-expressed `PRDM16` places eleventh.

> Both commands take **one** library. That is a correctness constraint, not a convenience: a belief
> about library *L* is emitted by the fold that excluded *L*, so arms in different libraries are
> expressed in different fitted bases and their coordinates are not comparable. The signature makes
> the incomparable case impossible to write.

> `sd >=` is a declared **lower** bound. It adds the two posterior variances as though the arms were
> independent, and they are not — both were scored under the same fold. The properly grouped
> interval is what `measure` reports.

Then ask whether that ordering is anything:

```bash
uv run python scripts/explore.py ranks
```

`rep1`'s ordering is easy to read as a result — the two master erythroid/MK regulators do come out
on top there. Across all fourteen libraries it does not replicate:

| target | mean rank | best | worst | NT CPM |
|---|---:|---:|---:|---:|
| RUNX1 | 6.0 | 1 | 16 | 10,020.9 |
| MYB | 6.4 | 1 | 19 | 3,139.3 |
| GATA1 | 7.3 | 1 | 18 | 2,045.4 |
| … | | | | |
| PRDM16 | 11.7 | 3 | 17 | **3.5** |
| SNAI2 | 12.7 | **1** | 19 | **0.6** |

`GATA1` holds rank 1 in three of fourteen libraries and falls to eighteenth in another. Every target
spans most of the field. And decisively: **in `rep3`, `SNAI2` produces the largest contrast of all
nineteen targets** — larger than `GATA1`'s 1.909 in `rep1`, the number the sanity check is built on.
`SNAI2` is at 0.6 CPM and cannot have been knocked down, so whatever ranked it first is what is
ranking everything else.

### 8. `measure` — the shipped capability measurements

```bash
uv run python scripts/explore.py measure
```

S2, S4 and S5, on held-out libraries, with cluster-bootstrap intervals grouped at the library.
All fail, and the block decomposition underneath says why:

```
  across-library variance in the NUISANCE block  :    81.3016   <- where library variation SHOULD land
  across-library variance in the BIOLOGY block   :     0.6087   <- leakage; S5's numerator
  between-target variance in the BIOLOGY block   :     0.1091   <- signal; S5's denominator
```

Read the terms, never the quotient alone. Library variation *is* being captured — 99.3% of it lands
in the nuisance block. S5 fails because its **denominator** is near zero. ADR 0022 cut the leakage
five-fold (3.07 → 0.609), a large real improvement in exactly what S5 names, and S5 barely moved
because the signal fell with it (0.257 → 0.109).

### 9. `calibration` — S6, and why a ratio was hiding the diagnosis

```bash
uv run python scripts/explore.py calibration
```

The only readiness criterion this backend evaluates. It scores the *same* evidence S2 does — ADR
0023's split-half replicate — but counts coverage instead of aggregating into a ratio, and the two
disagree. Per [ADR 0025](../adr/0025-s6-is-gated-on-the-whole-coherent-nominal-interval.md) it is
gated at **every** nominal the predeclared pair is coherent on, not one:

```
    nominal  coverage    error    BOUND   verdict
       0.90    0.8836   0.0164   0.0548   failed  <- reference; the belief publishes this row
       0.91    0.8886   0.0214   0.0590   failed
        ...
       0.95    0.9093   0.0407   0.0767   failed
```

**The verdict is the conjunction; the reference row is only what the belief publishes.** At that row
the point estimate passes both thresholds and the bound does not — a criterion reporting its point
estimate would have called this calibrated (ADR 0015).

**Why six and not one.** ADR 0024 gated at 0.90 believing the thresholds forced it. They do not:
they are coherent on [0.90, 0.95], and 0.90 is the *loosest* member since the bound rises
monotonically. That matters because **a one-level gate is clearable by a constant** — multiplying
every predictive sd by any factor in [1.04, 1.21] clears 0.90, and 1.11 lands coverage on exactly
0.9000 with a bound of 0.0368, better than the shipped 0.0548. Across all six, one scalar clears
every level. One level tests scale; scale is free. Six test shape.

Two decompositions ship with the number because a single coverage figure invites two wrong readings:

- **The failure is a tail, not a scale error.** Trim the worst 2% of the 1,400 gene-library outcomes
  and the standardized spread goes 1.2848 → 1.0045 — the spread is exactly earned. S2's 0.8415 reads
  as "uniformly 16% too narrow" and implies inflating ψ². That would push the already-conservative
  98% into over-coverage and still miss a 9.5σ outlier. **The two statistics point opposite ways.**
- **Coverage runs against depth**, 0.94 in the shallowest library to 0.76 in the deepest,
  `corr = −0.857`. That is the failure ψ² exists to prevent. ⚠️ Depth and differentiation day are
  collinear here and this design cannot separate them.

## What to do from here

The system is functional, runs on real data, and is iterable — the bar it was built to. What it is
**not** is validated: the ledger is 0 of 10 and it will stay 0 of 10 on this substrate, because the
perturbation arm is a measured null.

Two consequences worth internalising before proposing work:

- **Hardening the instrument cannot move the ledger.** ψ², the S2 estimand and the admission gates
  have all been repaired, and none of it moved a capability measurement. The claim that none of
  it *can* — that a better instrument aimed at no signal returns a better-characterised zero —
  rested on the withdrawn null verdict and no longer stands on its own.
- **Screen the substrate first — but not with this screen.** `knockdown` has the *shape* of a
  pre-download gate, and that is exactly how it misled: on-target transcript is a valid check
  only for a modality that acts on transcription. Record the perturbation modality before
  choosing the screen, and prefer controls that hold across modalities: guide-level
  replication, expression-dependence of effect size, and cutting-versus-non-cutting.

## Interactive use

Everything the tool prints is available from a REPL:

```python
from cellstate.backends.gse274113 import (
    available_arms,
    estimate_arm,
    describe_state,
    compare_arms,
    load_arm_slice,
)
from cellstate.backends.gse274113.fit import fit_fold

slice_data = load_arm_slice()
fold = fit_fold(slice_data, "rep1")  # the fold that never saw rep1
belief = estimate_arm("rep1", "GATA1")  # a typed CellStateBelief
print(describe_state(belief))
```

> ⚠️ `load_arm_slice` is cached and returns a **shared** `ArmSlice`. The dataclass is frozen but the
> count arrays inside it are not — mutating one in a REPL silently changes every later call,
> including `measure`. Copy before you edit: `slice_data.counts[key].copy()`.
