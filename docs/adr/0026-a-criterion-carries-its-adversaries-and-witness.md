# ADR 0026: A criterion carries the adversaries it must refuse and the witness it must pass

- **Status:** Proposed
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

Two criteria have been measured to be passable **by constructions that read no evidence**.

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

### What the two families have in common

Neither is a bug in an implementation. Both are properties of the criteria **as posed**, and neither
was discovered by reading the code. Both were found by attempting to clear the gate with a
construction chosen to be worthless, which is a thing no test in this repository does.

## Decision

1. **A criterion declares, in code, the adversaries it must refuse.** An adversary is an executable
   construction, chosen to be scientifically worthless, that the criterion must not report as
   passing. The founding entries are the two families above; the list is expected to grow.

   **Each adversary is recorded with the value it produced while the criterion still accepted it.**
   Without that, "declares an adversary" is satisfiable by a construction the criterion was never in
   danger of passing, and the requirement becomes a formality that certifies nothing — which is the
   defect shape this record exists to close, reproduced inside its own remedy. The tables above are
   that record for the founding two: S5 at 1.51e-08 passing, S2 at 1.3676 passing. An adversary that
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

3. **A criterion with no witness may not emit a verdict.** It reports `UNINTERPRETABLE`. This is a
   property of the type rather than a column somebody maintains, so a criterion cannot be read as
   negative merely because nobody has established that it could ever be positive.

   This does not weaken rule 10. A negative result still graduates its phase; what changes is that a
   result must first be *interpretable* to be negative.

4. **No absolute floor is adopted for S5.** Decision 1 supersedes the mechanism the earlier draft
   proposed. Any future refusal threshold on a scale-invariant statistic must itself be
   scale-invariant, and must be accompanied by an adversary demonstrating what it closes.

5. **No published value changes.** S2, S4, S5 and S6 keep their measured numbers. What changes is
   what may be concluded from them, and which of them may be quoted as verdicts at all.

## Consequences

Every existing criterion is `UNINTERPRETABLE` on adoption except S6, and stays so until it carries a
witness. That is an accurate description of what this repository currently knows, and it is a
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
