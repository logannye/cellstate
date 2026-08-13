# ADR 0019: Build the representation on evidence already held, ahead of freezing an estimand

- **Status:** Accepted; the queue mapping in decision 1 is **superseded** by
  [ADR 0020](0020-rna-first-nuisance-separation.md)
- **Date:** 2026-08-13

> **Read the ordinals through ADR 0020.** This record's decision is unchanged — the representation is
> built on held evidence before an estimand is frozen — but ADR 0020 inserted the held-out-modality
> test as its own item, which shifted every ordinal after it. Where this record says `Q6` (first
> belief) read `Q7`; where it says `Q7` (freeze the estimand) read `Q8`; where it says `Q8` (state
> backend) read `Q9`. The queue in `docs/roadmap.md` is the authority.

## Context

[ADR 0018](0018-gse274113-rejected-for-the-state-estimand.md) left Phase 2 with no candidate source
and the program with **zero registered sources satisfying S1**. `Q5` — freeze the state-bearing
estimand — cannot start, and its unblocking depends on finding an experiment that already happened
and has the right shape. That search has no schedule and no guaranteed end.

Behind that blocked item sit two that are **not** blocked. `Q6` fits observation models (S5). `Q7`
emits the first biological belief (S2). Neither needs a unit that spans an inference cutoff, neither
needs a second horizon, and both can run on evidence this project already holds and has already
reviewed.

Meanwhile the state-capability ledger stands at **0 of 10**, and the reason is structural rather
than incidental. Measured over the repository: `CellStateBelief` is constructed at exactly **one**
site in `src/`, a synthetic linear-Gaussian reference. Every scheduled item so far — the metric
suite, the interval estimator, both faithfulness harnesses, the fail-closed repair — has built the
apparatus that *judges* a representation. **The thing being judged has never been built.** A project
whose purpose is to compute a faithful representation of cellular state has, to date, computed no
representation of cellular state.

Continuing to hold `Q6` and `Q7` behind a blocked `Q5` means that state persists for as long as the
source search does.

### What the held evidence can actually carry

`GSE274113` was rejected for the state estimand because no library spans a timepoint, the longest
horizon is empty, and above the library it supplies one independent parent culture. None of those
three facts bears on S2, S4, or S5, because none of those capabilities requires a spanning unit, a
horizon, or a between-donor bootstrap.

- **S5** — the library nuisance axis is measured: across-library spread at a fixed target is 0.53,
  0.51, 0.89 and 0.99 of across-target spread at days 7, 9, 11 and 14. S5 asks that varying a
  nuisance variable at fixed biology move the predicted *observation* and not the inferred *state*.
  A nuisance axis of that size is the material the test needs. The property that disqualified the
  source for one claim is what qualifies it for this one, exactly as rule 7 contemplates.
- **S2** — a posterior whose spread is earned, judged on held-out libraries. Needs units, not time.
- **S4** — newly reachable, and previously not reachable at all. The series carries `NT` as a
  **declared-null** intervention against 19 perturbed transcription factors, with all 280 of 280
  `(library, target)` combinations populated, so intervention is fully crossed with library. That
  supplies both halves of S4's contrast: the null arm must leave the predictive distribution
  unchanged to numerical tolerance, and a non-null arm must change it by more than between-seed
  variation.

S4's null half deserves particular note. Nothing in this repository currently asserts it. The
existing reference test asserts only that a non-null intervention changes the prediction at all,
which is not the same as changing it by more than between-seed variation, and says nothing about the
null arm. **An inert `do` operator, or one that moves spuriously, passes every test this project
has.** That is the same class of defect as the one ADR 0017 closed: a check whose failing branch
cannot be reached.

## Decision

1. **Reorder the queue so the representation is built on held evidence before an estimand is
   frozen.** The observation-model item and the first-belief item move ahead of the estimand-freeze
   item. No item is added, removed, or rescoped beyond the additions in decisions 2 and 3; this is
   an ordering change, which is why rule 4 fires and why this record exists.

   | Before | After | Note |
   | --- | --- | --- |
   | `Q5` freeze the estimand | `Q7` | unchanged in substance; blocked on source selection |
   | `Q6` observation models | `Q5` | gains its source and its predeclared-bound condition |
   | `Q7` posterior inference | `Q6` | gains S4 and an explicit S2 spread condition |
   | `Q8` state backend and verdict | `Q8` | unchanged |

2. **`Q5` is bound to `GSE274113`,** the source ADR 0018 rejected for the estimand and retained
   under rule 7 for this. Its licence terms and byte identity are preconditions of the item, not
   follow-ups: rule 6 requires resolved provenance before bytes are used as evidence, and ADR 0018
   recorded both as open.

3. **`Q6` gains S4, and both halves of S4 are required.** A report that exercises only the non-null
   arm does not satisfy it. The null arm — a declared-null intervention leaving the predictive
   distribution unchanged to numerical tolerance — is the half that can currently pass by accident,
   so it is named explicitly in the item's done-when.

4. **This ordering does not weaken the sufficiency verdict, and does not substitute for it.** S7
   remains the definition of faithful and remains unreachable on held evidence. Nothing produced
   under `Q5` or `Q6` may be described as a sufficiency result, a faithfulness verdict, or evidence
   that the state is complete. What they produce is a representation that exists, emits through the
   public contract, and has been tested on three capabilities that do not require a spanning unit.

5. **Source selection continues in parallel and is not descheduled.** `Q7` stays in the queue with
   its blocker named. Building on held evidence buys the ability to build; it does not buy S1, and
   no amount of work on S2, S4 or S5 will produce S1.

## The honest cost

Three capabilities advanced without S1 is not a faithful representation. A belief can be
well-calibrated, carry an earned posterior, respond correctly to interventions, and separate
nuisance from biology, while still being a *description of a snapshot* rather than a state that
predicts a future. The ledger says so directly: a query for which S1 or S3 is structurally
unavailable cannot test S7.

So this record accepts a real risk — that a working prototype which advances 3 of 10 capabilities
reads as more progress toward the purpose than it is, and that momentum on the reachable capabilities
crowds out the unreachable one that actually defines success. Decisions 4 and 5 exist to hold that
line, and they are the parts of this record most likely to be quietly eroded.

The alternative was to build nothing until an S1-bearing source is found. That is not the more
rigorous choice; it is the choice that leaves the project's central object unbuilt and its apparatus
untested against anything real, for an unbounded period, on the hope of a dataset.

## Consequences

- The next implementation item needs no new data, no access decision, and no new authorization
  beyond this record.
- `CellStateBelief` acquires its first non-synthetic construction site, and the public API acquires
  its first real caller.
- S4's null half becomes testable, closing a gate that could not previously fail.
- The ledger moves off 0 of 10, with the ceiling stated: S1, S3 and S7 stay out of reach on this
  evidence, and `Q7` remains the only path to them.
