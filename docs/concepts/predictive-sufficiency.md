# Predictive sufficiency

This is the definition of "faithful". A belief is faithful for a query when it is **predictively
sufficient** for that query's declared targets and its predictive distributions are **calibrated**.
Both are numeric verdicts with a sampling distribution. Neither is a judgment about the model's
architecture, and neither can be replaced by the other.

This file is the canonical statement of the sufficiency test. Other documents summarize it; where
they disagree, this file wins.

## What is being asked

The desired representation is the smallest one that preserves the distribution of future observations
and functional behavior under the query's interventions, environments, and horizons. There is no
universal minimal state: the same cells carry a different sufficient state for a ten-minute signaling
query than for a fourteen-day differentiation query.

## The test

Fit two predictors of the same held-out future target, with declared equal capacity — one given the
belief alone, one given the belief plus the raw evidence history — and compare a predeclared proper
score:

```text
M1: Z_{t+h} = f(B_t^Q, U, E, Q)
M2: Z_{t+h} = f(B_t^Q, H_{<=t}, U, E, Q)
gain = score(M1) - score(M2)
```

Four properties of that comparison are not optional.

1. **The score is negatively oriented.** It is a loss — CRPS, energy score, negative log predictive
   density — so lower is better and the history information gain is nonnegative in expectation. A
   negative sampled gain is estimation noise, not evidence that raw history hurts, and must never be
   read as margin.
2. **The verdict is the interval, not the point.** The gain is reported with a bootstrap confidence
   interval resampled at the declared independent experimental unit — well, plate, library, donor,
   clone, or study — never at the cell. The belief is sufficient for `Q` when the upper end of that
   interval falls below the query's declared tolerance
   (`AcceptanceThresholds.maximum_history_information_gain`). **A gain reported without an interval is
   not a verdict.**
3. **The future must be genuinely held out, and `M2` must see the raw history.** Reconstruction of
   the present assay cannot substitute for prediction of a held-out future, and a summary of the
   history in place of the history makes the comparison uninformative in the safe direction.
4. **The harness must be null-calibrated before its verdicts are trusted.** It must return the right
   answer on a design where the state is sufficient by construction and on one where it is not.

## When the test does not apply

`M1` and `M2` differ only through the raw history. A query with no admissible observation before the
inference cutoff has an empty history, `M2` is identical to `M1`, and the comparison is vacuous — a
passing number that measures nothing. A single horizon likewise cannot distinguish a state from a
condition label. These are properties of the experiment, not of the software, and no amount of
contract rigor repairs them. Capabilities S1 and S3 of the
[state-capability ledger](../roadmap.md#the-state-capability-ledger) exist to establish them before a
model is fitted.

## Reading a failure

A gain whose interval clears the tolerance means the belief omitted historically relevant
information. The response is scientific rather than cosmetic: investigate exposure duration,
divisions, parent phenotype, accumulated stress, prior signaling pulses, chromatin, and
repeated-treatment history, then either expand the state factors or narrow the query. A sufficiency
test that fails, reported with its interval, is a result. Suppressing it is not.

## The companion test

Sufficiency alone is not faithfulness. Nominal predictive intervals must attain nominal coverage on
held-out units, reported as an absolute coverage error with an **upper confidence bound**, grouped at
the same split unit and compared to the query's declared threshold. A belief that is sufficient but
miscalibrated is unusable; one that is calibrated but insufficient is calibrated about the wrong
thing.

Cluster coherence, reconstruction error, and attractive low-dimensional projections are not
substitutes for either test.

## Implementation status

`evaluate_history_information_gain` (`src/cellstate/evaluation/sufficiency.py`) computes
`gain = state_only_loss - state_plus_history_loss` and compares it to the query threshold. It accepts
two losses a caller must already have produced: it does not fit `M1` or `M2`, does not resample, and
returns no interval. `SufficiencyReport` has no interval field. `empirical_interval_coverage`
(`src/cellstate/evaluation/calibration.py`) returns a coverage fraction with no upper bound. Neither
function has a caller outside `tests/`.

`markov_sufficiency_score` is `exp(-max(gain, 0))` — a monotone restatement of the gain, not
independent evidence, and it must not be reported as a second measurement.

Building both harnesses, and the serialized-contract changes they require, is Phase 1 of
[`../roadmap.md`](../roadmap.md), which is the sole authority for when that happens. Until it lands,
this document states a definition the repository can express and cannot yet execute.
