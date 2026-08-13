# ADR 0017: The sufficiency verdict must fail closed, and the floor moves onto the state estimand

- **Status:** Accepted
- **Date:** 2026-08-13

## Context

[ADR 0016](0016-the-verdict-gates-on-the-interval.md) closed three defects that shared one theme: a
quantity reported with a sampling distribution attached, where the distribution bound nothing. This
record closes a fourth defect of the same family and, having done so, reorders the queue that the
defect makes unrunnable.

### 1. The test that defines the project's purpose passes when it has nothing to test

`evaluation/sufficiency.py` compares `M1` (state alone) against `M2` (state plus raw pre-cutoff
history) and reports the history information gain. Approximate sufficiency is supported when the
gain's interval upper end falls within the tolerance.

Measured on a design constructed so that history genuinely drives the target — the harness returns
`gain = 19.7297`, interval `[13.1951, 29.7008]`, `FAILED`, which is correct — with every history
block replaced by zeros:

| History block | Gain | Interval | Outcome |
| --- | --- | --- | --- |
| real | 19.7297 | `[13.1951, 29.7008]` | `FAILED` |
| all zeros | **0.0000** | **`[0.0000, 0.0000]`** | **`PASSED`** |

A query carrying no admissible pre-cutoff evidence therefore receives not a weak certificate of
sufficiency but the **strongest one the contract can express**: a gain of exactly zero with a
degenerate interval. ADR 0016's own change deepens this. Having correctly moved the gate onto the
interval's upper end so that imprecision could no longer be read as sufficiency, it left a case where
the interval is not merely narrow but a point — maximal confidence, earned by the absence of
evidence.

The module knew. Its docstring says that a query with no admissible pre-cutoff evidence "makes `M2`
identical to `M1` and the test **inapplicable rather than passed**; that judgment belongs to the
query and its benchmark, not to this module."

The delegation is void, because there is no delegate. `grep -rn evaluate_predictive_sufficiency
src/ scripts/ examples/` returns only the defining line and re-exports from
`evaluation/__init__.py`. No query, no benchmark, and no bundle calls this function. The judgment was
assigned to a caller that does not exist, so nothing performs it, and the harness's own answer —
`PASSED` — stands unchallenged. This is a check whose false branch is safe.

### 2. `Q2` was recorded as delivered against a completion condition it does not meet

`Q2`'s done-when reads: *"both functions have non-test callers, both reports carry intervals, and the
harness returns the correct verdict on a sufficient and an insufficient synthetic design."* The
second and third clauses hold. **The first does not**, by the grep above.

The roadmap's Current Status nonetheless records `Q1` and `Q2` as delivered and asserts that "the
project can recognize a faithful representation, which it could not before." On the evidence of
defect 1, and with no caller to perform the applicability judgment, that sentence claims more than
the code earns. Recording it was an error, and this record corrects it rather than quietly restating
it.

The two defects are the same defect. A harness with no caller has no one to refuse an inapplicable
design, and a harness that cannot refuse produces a verdict that means nothing when it is most
confident.

### 3. `Q3` measures a floor on a query that is not on the state path, and is blocked twice

`Q3` measures the observational floor on the sci-Plex3 population-response component. That component
is already classified in this roadmap's Current Status as unable to satisfy S1, S3, or S7 and as "not
a step toward the purpose." `Q5` freezes a different estimand, which will carry its own baselines and
its own floor, so a floor measured at the proving ground does not transfer to the query that needs
one.

It is also blocked twice rather than once. The recorded blocker is the ADR 0011 access grant, whose
issuing control plane is suspended. The second was found on first contact with the bytes: the frozen
`zero_panel_total_policy` is `error_fail_evaluation_no_exclusion_or_imputation`, `p1-train` contains
seven records whose 2,000-feature panel total is exactly zero, and the policy forbids the only three
responses that would let the evaluation proceed. Granting access would not unblock the item.

`Q3`'s own text anticipates the exit: the required ADR may "authorize a single held-out read for
baseline-versus-baseline scoring" **or** "move the floor measurement onto the Phase 2 estimand." This
record takes the second branch.

## Decision

1. **The sufficiency harness refuses an inapplicable design; it does not pass it.** The applicability
   judgment moves out of the docstring and into the module. A comparison in which the history block
   carries no information the state does not already carry is not evidence of sufficiency, and the
   contract must be unable to express it as such. `EvaluationStatus` already distinguishes evaluated
   from not-evaluated; an inapplicable design resolves there, never to `PASSED`.

2. **Units lacking a pre-cutoff observation are excluded from the paired comparison, the retained
   fraction is a required field on the report, and a retained fraction of zero is a refusal.** A unit
   with an absent history block is not a unit on which the question was asked. Diluting the mean with
   such units moves the verdict toward `PASSED` and, because their per-unit difference carries no
   spread, narrows the interval at the same time — bias and false precision from one cause. The
   endpoint of that dilution is the table above.

3. **`Q3` is retired, not deferred, and its floor measurement is absorbed into `Q5`.** The queue slot
   is reused for the repair decisions 1 and 2 authorize, because that repair is what actually blocks
   S7. Queue IDs are ordinals within the current queue and not stable identifiers (rule 3), so the
   mapping is recorded below rather than worked around.

4. **Phase 1's third graduation-gate bullet is restated.** It required a scoreboard in which every
   applicable baseline is scored against every other on a real held-out partition of the proving
   ground. Decision 3 makes that unreachable by design rather than by delay, so continuing to carry
   it would leave Phase 1 with a gate that cannot be passed and no record of why. It is replaced by
   the requirement that the floor be measured on the estimand `Q5` freezes, with the same
   interval discipline. This is a graduation-gate change and is the reason rule 4 fires here.

5. **ADR 0014's decision 5 is re-homed.** The decision whether to publish `benchmark_version` `1.1.0`
   of the frozen artifact with executable metric bindings was assigned to `Q3` by name. Retiring
   `Q3` would orphan it. It moves to `Q5`, which becomes the first item that runs executable metrics
   against frozen partitions.

6. **Two constraints carry into `Q5`, both earned on real bytes.** First, the **zero-panel
   doctrine**: a sample whose panel total is zero is a missing observation under an observation
   model, not an evaluable target, and no suite frozen from here on may adopt a policy that makes a
   benchmark unevaluable on its own data. Second, the **reachable-threshold constraint**: no metric
   may be frozen with an acceptance threshold that has not been demonstrated reachable on real bytes.
   The first run measured 94.63% of transformed coordinates in a well to be exactly zero, which makes
   marginal coverage an estimate of panel sparsity — empirical coverage moves only from 0.948 to
   0.975 across nominal 0.50 to 0.95, so two of the three frozen coverage metrics cannot meet their
   0.03 bound for reasons no candidate can affect, and the third passes uninformatively.

7. **Current Status is corrected** to record `Q2` as incomplete against its own done-when, and to
   withdraw the claim that the project can recognize a faithful representation. It can compute a
   gain with an interval. Recognizing faithfulness additionally requires refusing the designs on
   which the question is not being asked, which is what decisions 1 and 2 authorize.

8. **The serialized contract gains a field and keeps its version.** Decision 2 adds
   `retained_unit_fraction` to `SufficiencyReport`. Phase 1 requires a schema-version decision for
   any such change, so this is it: the version stays **`2.0`**.

   The field is additive, optional, and defaulted, so nothing about the wire format breaks. What
   does change is which documents are *constructible*: an evaluated report that omits the fraction,
   and any report claiming a fraction of zero, become unrepresentable. That is a real tightening of
   what a producer must supply, and calling it backward-compatible would be wrong.

   It is nonetheless kept at `2.0` because `v2` has no external producers to break — no biological
   backend is registered and no benchmark is scientifically admitted, so every producer of a
   `SufficiencyReport` is inside this repository. **That reason expires the moment one exists.**
   The next tightening of an evaluated report's obligations, once anything outside this repository
   emits one, is a version bump and not a judgement call. ADR 0016 kept `2.0` through a larger
   change — the meaning of the outcome itself — on the same reasoning, and this record is
   deliberately consistent with it rather than quietly stricter.

### Queue mapping

| Before | After | Note |
| --- | --- | --- |
| `Q1` metric suite | `Q1` | unchanged, delivered |
| `Q2` faithfulness harnesses | `Q2` | unchanged in scope; recorded incomplete against its own done-when |
| `Q3` measure the observational floor | *retired* | absorbed into `Q5` per decision 3 |
| — | `Q3` make the sufficiency verdict fail closed | new, authorized by decisions 1, 2 and 7 |
| `Q4` review and manifest the source | `Q4` | unchanged |
| `Q5` freeze the estimand | `Q5` | gains the floor measurement and decisions 5 and 6 |
| `Q6`–`Q8` | `Q6`–`Q8` | unchanged |

External documents cite artifacts and ADRs rather than queue IDs (rule 3). Where existing records
cite `Q3`, they refer to the retired floor item and are read through this table.

## What this record deliberately does not decide

**Whether the reported gain requires a noise-share decomposition, and by what construction.** If the
pre-cutoff observation is a second noisy read of the same quantity the state summarizes, `M2` can
beat `M1` by error-averaging alone, and the across-unit permutation cannot separate that from
dynamics because error-averaging *is* unit-specific association. A replicate control — substituting a
contemporaneous same-condition replicate for the history block and reporting the difference of gains
— was considered and is **not adopted**. It is not prototypable on any data this project holds:
`benchmarks/vertical-a/sciplex3-k562-24h-v1` maps 188 compounds at four doses to 752 treated wells in
`p1-train`, an exact bijection, so no treated condition has a second contemporaneous well. Whether a
different construction is warranted is an empirical question, and the measurement that settles it —
split-half reliability of the state and history blocks on real bytes — needs no data access and no
further authorization. It is scheduled as evidence, not decided here.

**The disposition of `GSE274113` and the Phase 2 source.** The structural census
([`../data/representability/gse274113-perturb-multiome.md`](../data/representability/gse274113-perturb-multiome.md))
establishes that the intended design is not realizable, but `Q4`'s done-when requires a reviewed
manifest carrying licence terms, byte identity against the GEO record, and the independent
parent-culture count. The last of these decides whether the measured ρ = −0.026 at day 7 → day 14 is
a decay finding or a culture confound, and it sets the source's maximum achievable unit count. A
source decision taken before those facts would be taken on assumption, which is the failure the
census exists to prevent. It belongs in its own record, after `Q4`.

## Consequences

The project loses its nearest available real-data number. The proving-ground scoreboard was the
cheapest scientific output on the board and it is now off the board; the first floor this project
measures will be on the Phase 2 estimand, later and at greater cost.

That is the correct trade. A floor on a query that cannot carry S1, S3, or S7 cannot become a
sufficiency verdict, and the item was unrunnable in any case. Against it, the repair moves the
project from a test that returns a confident `PASSED` on absent evidence to one that refuses — which
is the difference between a system that can recognize a faithful representation and one that cannot
recognize an unfaithful one.

Rule 2 applies: this record and the queue amendment it authorizes land together and alone, and the
implementation follows as a separate change.
