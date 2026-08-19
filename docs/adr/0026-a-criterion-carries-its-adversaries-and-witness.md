# ADR 0026: A criterion carries the adversaries it must refuse and the witness it must pass

- **Status:** Accepted
- **Date:** 2026-08-19

## Context

Nine of this repository's ten ledger criteria have never been observed passing on any substrate,
real or synthetic. The single exception is S6, whose `test_the_six_level_gate_can_be_passed`
rescales the residuals by 1.21 and observes all six nominal levels clear, with 1.20 and 1.22 failing
on opposite sides.

The consequence is that `0 of 10` cannot be attributed. A failing rung is consistent with a null
substrate and equally consistent with a criterion nothing could pass, and the project holds no
measurement that separates them. This is not hypothetical: the substrate half of that ambiguity was
believed for weeks and has since been withdrawn (PR #41), and the deposit is now measured to carry
target-consistent structure at 2.66x a within-library permutation null, p = 5.0e-4.

Four families of construction have been measured to pass criteria they do not estimate. Two of
them take **three ledger rows between them**, need no change to the estimator at all, and would
plausibly be written by an implementer acting in good faith.

### The founding adversary: split the count vector instead of the cells

Three lines applied to the deposited payload, and **three ledger rows flip from FAIL to PASS**:

```python
for lib in payload["libraries"]:
    nt = np.asarray(arms[(lib, "NT")]["counts"], dtype=np.int64)
    arms[(lib, "NT_A")]["counts"] = (nt // 2).tolist()
    arms[(lib, "NT_B")]["counts"] = (nt - nt // 2).tolist()
```

| measurement | as shipped | under the adversary |
| --- | --- | --- |
| S2 earned spread | 0.84148 [0.71437, 0.97593] FAILS | **1.22737 [1.03663, 1.43767] PASSES** |
| S4 null half | 2.02578 [1.43838, 2.67070] FAILS | **0.64356 [0.47905, 0.80402] PASSES** |
| S4 non-null half | 2.09008 [1.61546, 2.56195] FAILS | **2.10127 [1.63093, 2.56178] PASSES** |

It is worthless because both halves are **the same cells counted twice**. A split-half replicate
exists to expose a model to sampling variation it has not seen; halving a count vector produces two
vectors with no replicate variance at all, whose entire difference is integer parity. S2 then
measures how well an eight-column design fits one vector twice, and S4's null half measures the
behaviour of integer division.

**It is invisible to every check this repository ships.** `NT == NT_A + NT_B` still holds bitwise, so
the integrity assertion at `tests/test_gse274113_observation_model.py:891` passes unchanged. The
documented exact-2.00x depth shortfall still holds. There is no zero-width interval to notice. That
assertion's own comment states the correct invariant -- *"splits the NT cells, it does not
resample"* -- and then checks a weaker one the adversary satisfies, which is this repository's
recurring defect in miniature: the claim was written somewhere that could not check it.

**It is could-ship-by-accident, which is what makes it the founding entry.** "Split-half" is a
natural thing to read as splitting the vector. Anyone reimplementing the placebo split under
pressure to make S2 pass would very likely write exactly this, and every guard would agree.

**The refusal it implies is available and cheap.** `ArmSlice` already carries `cells`, and on the
real slice `cells[NT_A] + cells[NT_B] == cells[NT]` in all fourteen libraries with the halves
genuinely unequal (484/472, 361/347). A count split says nothing about how many cells produced it,
so asserting the partition at the **cell** level catches it where asserting it at the count level
cannot.

### The pooling family: discard the variable the criterion measures

One overridden method, no tuning parameter, no floor:

```python
class Pseudobulk(ArmSlice):  # "merge the replicate libraries per target"
    def log_composition(self, library, target):
        libs = [l for l in self.libraries if (l, target) in self.counts]
        comp, _ = log_composition(np.sum([self.counts[(l, target)] for l in libs], axis=0))
        return comp, float(self.counts[(library, target)].sum())
```

| measurement | as shipped | pooled |
| --- | --- | --- |
| S5 nuisance separation | 10.36468 [6.26717, 16.65935] FAILS | **0.00042 [0.00022, 0.00065] PASSES** |
| S4 separation | null 2.026 vs perturbed 2.090, not separated | **null 1.013 [0.985, 1.044] vs perturbed 1.667 [1.660, 1.675], separated** |

S5 clears its bound by a factor of eight hundred. It is worthless because after pooling,
`log_composition` is a function of the **target alone** -- library identity is discarded before the
model sees it -- and S5's estimand *is* across-library spread at fixed target. The criterion is
cleared by deleting the quantity it exists to measure. It also silently breaks leave-one-library-out:
the pooled vector handed to the fold that excluded `rep1` contains `rep1`.

### S5 — the label-function family

`measure_nuisance_separation` divides each arm's deviation from its own target's mean state by the
across-target variance. A state that is a deterministic function of the target label makes every
deviation zero. Measured on seventy synthetic arms across fourteen libraries:

    state                          within-target msd    S5          interval width   passes
    f(label)                       3.56e-32             6.45e-32    0                yes
    f(label) + 1e-6 * N(0,1)       8.35e-13             1.51e-12    non-zero         yes
    f(label) + 1e-4 * N(0,1)       8.35e-09             1.51e-08    non-zero         yes

The real estimator, on real evidence, returns **10.365 [6.27, 16.66] and fails**. The gate's unique
optimum is therefore an estimator that reads nothing.

**An earlier draft of this record proposed an absolute floor of `1e-9` on the within-target spread.
That floor is defeated by the third row above, which clears it by 8.5x and also loses the zero-width
interval that made the first row visible.** The error is dimensional and worth stating so it is not
repeated: **S5 is a ratio and therefore scale-invariant** — multiply every biology coefficient by a
thousand and S5 is unchanged while the within-target mean squared deviation scales by a million. An
absolute floor on a scale-invariant statistic's numerator cannot mean anything.

### S2 — the outside-the-estimand family

S2 is a split-half replicate on `NT` (ADR 0023). Its estimand looks only at `NT`, `NT_A` and `NT_B`.
Measured on the committed slice, adding gene-wise multiplicative noise to the **targeted** arms only,
leaving all three `NT` arms bit-identical:

    sigma       S2        lower     passes
    0.00        0.8415    0.7144    no        <- the committed slice
    0.25        1.0140    0.8677    no
    0.50        1.3676    1.1798    YES

**S2 is passed by degrading arms its own estimand never examines.** The mechanism is that `psi^2` is
one global scalar fitted across all arms (`fit.py`), which enters the numerator through
`fold.observation_variance` and never the denominator. No change to the estimator is required — only
noisier data elsewhere.

### What the four families have in common

None is a bug in an implementation. All four are properties of the criteria **as posed**, and none
was discovered by reading the code. All four were found by attempting to clear a gate with a
construction chosen to be worthless, which is a thing no test in this repository does. Two of them
require no change to the estimator whatsoever -- only to the data it is handed.

## Decision

1. **A criterion declares, in code, the adversaries it must refuse.** An adversary is an executable
   construction, chosen to be scientifically worthless, that the criterion must not report as
   passing. The founding entries are the four families above; the list is expected to grow.

   **Each adversary is recorded with the value it produced while the criterion still accepted it.**
   Without that, "declares an adversary" is satisfiable by a construction the criterion was never in
   danger of passing, and the requirement becomes a formality that certifies nothing — which is the
   defect shape this record exists to close, reproduced inside its own remedy. The tables above are
   that record for all four: the count split at S2 1.22737 and S4 0.64356/2.10127, pooling at S5
   0.00042, the label function at S5 1.51e-08, and outside-estimand noise at S2 1.3676. An adversary that
   cannot be shown to have passed something is not evidence about a gate.

   **The list lives in code and not in this record, deliberately.** Accepted ADR bodies are never
   rewritten (`AGENTS.md`), so an adversary enumerated here could never be extended without a new
   decision record. A future adversary is a new entry in a registry and a new test, not an
   amendment. This record fixes the *requirement*, not the *membership*.

2. **A criterion declares the witness it must pass.** A witness is an executable construction on
   which the criterion is expected to pass, together with a neighbouring construction on which it
   fails. `tests/test_s6_calibration.py` is the reference implementation.

   A witness may be synthetic. It establishes a property of the **criterion** and is never
   biological evidence; no witness may be cited on the scoreboard, in a model card, or as support
   for any capability claim.

3. **A criterion without a witness keeps its verdict and carries its witness status beside it.**
   The ledger row reads `fails (no witness)`, not `uninterpretable`.

   An earlier draft of this decision replaced the verdict outright. That was worse, and the reason
   is worth recording: the measured value and the interval are real, were expensive to obtain, and
   are the only things that would let a future reader notice if a criterion started behaving
   differently. Discarding a verdict because its interpretation is limited throws away the
   measurement in order to express a caveat about it, when both fit on one line.

   What the qualifier forbids is precise and is the thing that actually went wrong here: **a
   `fails` without a witness may not be cited as evidence about the substrate, the deposit, or the
   model.** It is a measurement whose attribution is unestablished. This project spent weeks citing
   `0 of 10` as evidence that `GSE274113` carried no signal; the deposit is now measured to carry
   target-consistent structure at 2.66x a permutation null, p = 5.0e-4. That inference is what this
   clause bans, and banning the inference costs nothing that banning the verdict would have bought.

   Rule 10 is untouched: a negative result still graduates its phase.

4. **No absolute floor is adopted for S5, and no replacement is chosen here.** Decision 1 supersedes
   the mechanism the earlier draft proposed, and deliberately does not name a successor.

   Two candidates were considered and neither is adopted. A floor relative to `across_target` is
   scale-invariant but amounts to refusing a state for looking *too* separated, which is a strange
   thing to assert without knowing where the boundary lies. A floor relative to the estimator's own
   claimed posterior variance is stricter and catches a real incoherence -- a state that claims
   uncertainty it does not exhibit -- at the cost of coupling this criterion to the belief contract.

   **Neither has been tested against an adversary, and that is the whole reason to defer.** The
   `1e-9` floor in the earlier draft was also chosen by reasoning rather than measurement, and it
   was defeated by the third row of the table above. Picking a second untested threshold in the
   same record that documents the first one failing would be the same error with a different
   number. Under decision 1 any successor must arrive with the adversary it closes; until one does,
   S5 carries its adversaries and has no floor.

5. **No published value changes.** S2, S4, S5 and S6 keep their measured numbers. What changes is
   what may be concluded from them, and which of them may be quoted as verdicts at all.

## Consequences

Every existing criterion except S6 reads `fails (no witness)` on adoption, and stays so until it
carries one. That is an accurate description of what this repository currently knows, and it is a
temporary state whose exit is a day of work per criterion — the constructions for S4 (amplification)
and S2 (a placebo blend that leaves `psi^2` unmoved) are already identified.

**Source selection is blocked on decision 2 rather than on a new corpus.** Buying a deposit to test
an instrument that has never been shown to respond is the more expensive way to learn the same
thing, and the permutation screen has now removed the argument that the current deposit is empty.

Decision 2 may fail. If no witness can be constructed for a criterion, that criterion is malformed
as posed, and re-posing it is a larger decision than this record makes — it would need its own ADR
and it would outrank the remaining queue.

## What this record does not do

It does not repair S2's `psi^2` asymmetry, which is the mechanism behind its adversary family and is
a question about the observation variance rather than about the criterion. It does not decide
whether S5's refusal should be scale-relative to `across_target` or to the estimator's own posterior
variance; decision 1 requires only that whatever is adopted comes with the adversary it closes.
