# ADR 0014: Bind the Phase 1 metric condition to the frozen specification

- **Status:** Accepted
- **Date:** 2026-08-12

## Context

[ADR 0013](0013-state-first-roadmap-reordering.md) made Phase 1 first and queue item `Q1` its first
action. Both were given the same completion condition: no metric suite *frozen under this roadmap*
resolves any entry to `PortImplementationKind.SPECIFICATION_ONLY`.

That set is empty, and will stay empty for four more queue items.

One benchmark exists in this repository:
`vertical-a.sciplex3-k562-24h-replicate-transfer.v1`, `benchmark_version` `1.0.0`, `design_status`
`frozen`. It was frozen on 2026-08-09 by [ADR 0008](0008-sciplex3-k562-component-benchmark.md) —
three days before the roadmap that cites it — and ADR 0013 explicitly exempts it from retrofit. The
first suite that will be frozen under the current roadmap is the Phase 2 estimand's, at `Q5`, behind
a data acquisition that has not begun.

A condition quantified over an empty set is satisfied by construction. `Q1` and the first bullet of
the Phase 1 graduation gate are both true today, with no metric implementation written, and cannot
become false through any act of omission. ADR 0013 rejected declaring the sufficiency test satisfied
on a query where the history is empty, on the grounds that a gate that cannot fail is not a gate. The
same objection applies to its own first gate. The rule caught its author on its first use.

Second, the condition names the wrong artifacts. `Q1` lists five point metrics. The frozen suite
declares **ten** metric entries whose `implementation_binding.kind` is `specification_only`, and
**ten** `uncertainty.method` entries whose kind is also `specification_only` — one per metric, each a
`multiway_clustered` bootstrap over the `compound` and `plate` dependence units, `resample_count`
2,000, `confidence_level` 0.95. The uncertainty method is not an implementation detail of a metric.
It is what turns a number into a verdict, and the roadmap's first stated commitment is that a score
reported without an interval grouped at the declared independent unit is not a verdict. `Q2`
separately requires a multiway bootstrap grouped at the split unit. Naming it in neither item leaves
the shared substrate of both faithfulness tests unscheduled.

Third, a factual error carried by ADR 0013 and the roadmap alike. Both describe the frozen suite as
having "five metric families." It declares three `family` values — `calibration`,
`intervention_effect`, `predictive_proper_score` — across ten metrics. Neither number is five.

## Decision

1. **`Q1`'s completion condition binds to the frozen specification artifact.** The artifact is
   `sciplex3-frozen-metric-suite`, `sha256`
   `6f94fe0102f6e987cd5c5a1c6d31e58d5ad7c449c83e6d2b8e64196b38cf5634`, at
   `benchmarks/vertical-a/sciplex3-k562-24h-v1/support/metric-suite-spec.json`. `Q1` is done when
   every `metric_id` that specification declares resolves to an executable implementation with a
   golden fixture and an independently derived numerical reference.
2. **`Q1` delivers the multiway clustered bootstrap**, to the configuration the frozen suite
   declares. `Q2` consumes it rather than reimplementing it.
3. **The condition is mechanically enforced.** A conformance test reads the frozen specification and
   fails if any declared `metric_id`, or the declared uncertainty method, does not resolve to an
   implementation. The test verifies the specification's own `sha256` and byte count first, so that
   editing the specification to remove a metric cannot pass the test in place of implementing it.
   The condition is checked by the test suite, not asserted in prose. The on-disk file hashes to the
   value above at 4,664 bytes, matching the benchmark artifact's declaration, so the target is the
   frozen bytes and not a copy that has drifted from them.
4. **The first bullet of the Phase 1 graduation gate is restated to match**, and retains its
   forward-looking half: a suite frozen under this roadmap may contain no specification-only entry.
   That half is vacuous only until `Q5`, and is a constraint on `Q5` rather than on `Q1`.
5. **The frozen artifact does not change.** Its bindings remain `specification_only`, which is a true
   statement about a benchmark nothing has yet been scored against. Whether to publish
   `benchmark_version` `1.1.0` with executable bindings and `supersedes_version` `1.0.0` is a
   separate decision, and belongs to `Q3` — the first item that would run the implementations against
   the frozen partitions. Implementing a metric and re-freezing the benchmark that cites it are not
   the same act.
6. **The "five metric families" count is corrected** to three families across ten metrics, in this
   record, in the roadmap, and by marked amendment in ADR 0013.

## Consequences

- `Q1` fails today. `src/cellstate/evaluation/metrics.py` does not exist, so the conformance test
  decision 3 specifies will, once written, fail on all ten `metric_id` values it reads. That test is
  part of `Q1`'s implementation and not of this record. This is the intended state: the item's
  completion condition now distinguishes doing the work from not doing it.
- `Q1`'s scope grows by one estimator and one metric. The ten declared entries reduce to six distinct
  computations — sample CRPS, energy score, marginal coverage error, marginal interval width,
  vehicle-relative pseudobulk effect RMSE, and the equal-compound four-dose profile diagnostic — two
  of which are parameterized by nominal level and contribute three entries each. `Q1` as written
  named five of the six; the four-dose profile diagnostic was omitted, along with the bootstrap.
- The bootstrap is the first component in this repository whose correctness is a statistical property
  rather than an arithmetic one. Its golden fixture must pin a seeded resampling sequence, and its
  numerical reference must be a design whose interval is known analytically, not a recorded output of
  the implementation being tested.
- Nothing downstream of `Q1` is reordered. `Q2` through `Q8` keep their positions and their
  capability tags.
- This record contains no `src/` implementation, satisfying program rule 2.

## Rejected alternatives

- **Retrofit the frozen suite to executable bindings as part of `Q1`.** Re-versioning a
  content-addressed benchmark to assert that its metrics are executable, before any of them has
  scored a prediction on a real partition, records a claim ahead of its evidence. The binding kind
  should change when something has been run, not when code has been merged.
- **Freeze a new suite under this roadmap so the original condition becomes non-vacuous.** This
  inverts the dependency. It requires the Phase 2 estimand at `Q4`/`Q5` to precede the metric
  implementations, which is precisely the ordering ADR 0013 rejected — evidence acquisition ahead of
  the ability to recognize a result.
- **Leave the condition and rely on the deliverables list.** Phase 1's deliverables are unambiguous
  about what must be built. Its gate is not. Program rule 3 makes the queue the sole authority for
  status, so a gate that disagrees with the deliverables is the one that governs.
- **Delete the "frozen under this roadmap" clause entirely.** It has real work to do at `Q5`, where
  it forbids freezing a new suite with unimplemented entries. Removing it would drop a constraint
  that binds the next suite in exchange for repairing one that binds this one.
