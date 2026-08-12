# ADR 0015: The faithfulness reports carry their sampling distribution

- **Status:** Accepted
- **Date:** 2026-08-12

## Context

The roadmap's first commitment is that both faithfulness tests must return a numeric verdict with a
sampling distribution, and that a sufficiency gain or a coverage error reported without an interval
grouped at the declared independent experimental unit is not a verdict. Neither report can express
one.

`SufficiencyReport` carries `history_information_gain` and a threshold. `CalibrationReport` carries
`empirical_coverage`, `calibration_error`, and a threshold. Both validate an `outcome` against those
thresholds using point estimates alone. Capability S6 requires the coverage error to be within
threshold *as an upper confidence bound*, and S7 requires the gain to come *with a bootstrap
interval grouped at the split unit*. Neither is representable today, so a report that satisfied its
validator would still not be a verdict.

Queue item `Q2` therefore requires a serialized-contract change, and the roadmap requires that
change to come with a schema-version decision.

### What the version decision has to respect

The runtime version is a single global literal, `SchemaVersion = Literal["2.0"]`, carried by every
boundary object that has one. Two stored artifacts in this repository carry it, and both are frozen:

- `benchmarks/vertical-a/sciplex3-k562-24h-v1/state-query.json`, at its root;
- the same query embedded at `$.definition.query.state_query` of the benchmark artifact.

The query is not merely stored. Its bytes are pinned by `SCIPLEX3_K562_QUERY_SHA256`
(`d0fa67f31a8ea1d7b2e8839dfe7629fd6f359ea7eed4f6d336e2cd1d8813971e`) in `src/`, read through
`_read_exact`, validated as a `StateQuery`, and then compared byte for byte against
`canonical_json_bytes`. A change to the accepted set of version strings that did not include `"2.0"`
would fail that load outright.

`StateQuery` is an input contract. `SufficiencyReport` and `CalibrationReport` are output-side
objects nested in `BeliefDiagnostics`. No stored artifact in this repository contains a
`BeliefDiagnostics`, and no biological backend has emitted one, so no stored bytes change meaning
under this decision.

## Decision

1. **The runtime `SchemaVersion` stays `"2.0"`.** Advancing the global literal would relabel the
   frozen `StateQuery` input contract on account of a change that does not touch it — the silent
   relabeling [ADR 0005](0005-belief-subject-semantics.md) was written to forbid — and would break a
   hash pin to buy nothing, since no artifact carrying either report exists to be distinguished.
   The reports are self-describing instead: an evaluated report without an interval cannot be
   constructed after this record, so the presence of the field is the version signal.
2. **`SufficiencyReport` carries a grouped bootstrap interval** on the history information gain, and
   **an evaluated report must have one.** A report that reaches `EVALUATED` without a sampling
   distribution is rejected at construction. This is the enforcement of the roadmap's first
   commitment, in the one place that cannot be bypassed.
3. **`CalibrationReport` carries a coverage-error upper confidence bound**, and its outcome gates on
   that bound rather than on the point estimate. S6 asks whether the coverage error is within the
   threshold as an upper bound; a point estimate inside the threshold with a bound outside it is not
   a pass. An evaluated calibration report must carry the bound.
4. **The interval is a domain object.** `BootstrapInterval` moves from
   `cellstate.evaluation.bootstrap` to `cellstate.domain.common`, and the estimator imports it. A
   serialized contract belongs to the layer that owns serialization; `domain` must not depend on
   `evaluation`. The public name and the import path `cellstate.evaluation.BootstrapInterval` are
   unchanged.
5. **The paired predictors declare their capacity, and the harness enforces equality.** `M2` beating
   `M1` because it was allowed more parameters is a capacity effect, not history information. The
   harness refuses to run on predictors whose declared capacity differs, and the reference pair
   achieves equality by construction: both receive the same design matrix shape, and `M1` receives a
   permuted history block where `M2` receives the real one.

## Consequences

- `schemas/v2/` is regenerated. The emitted JSON Schemas gain the interval and bound; the version
  string in them does not change, which is the intended outcome of decision 1.
- Every existing construction of an evaluated `SufficiencyReport` or `CalibrationReport` must supply
  the new field. The synthetic reference backend and the contract tests are updated. This is a
  deliberate break of a shape nothing has emitted.
- `evaluate_history_information_gain` and `empirical_interval_coverage` gain callers in `src/` for
  the first time: the harnesses compute losses, coverage, and intervals, and delegate to them to
  build the report. They stop being reachable only from tests.
- The permuted-history design gives the null calibration for free. Under permutation the history
  block carries no information about the target while retaining its exact marginal distribution and
  its contribution to model capacity, so the expected gain is zero by construction and the interval's
  coverage of zero is measurable.
- Deciding not to advance the version is a decision that must not rot. The frozen query's load path
  already verifies its hash, its schema validity, and its canonical round-trip on every run, so a
  future change that would have required a bump fails there rather than passing silently.

## Rejected alternatives

- **Advance the global literal to `"2.1"`.** It relabels queries, histories, forecasts, and plans
  for a change confined to two output-side reports, and it breaks a hash-pinned frozen artifact.
- **Accept both `"2.0"` and `"2.1"`.** This keeps the frozen query loadable but makes the version
  string ambiguous about the report shape while adding a second value nothing distinguishes. A
  literal that admits two values so that one of them never appears in practice is decoration.
- **Fork `schemas/v2.1/` and freeze `schemas/v2/`.** [ADR 0005](0005-belief-subject-semantics.md)
  reserves a frozen directory for a breaking overhaul of the wire format, and the experimental
  manifests set the counter-precedent: ADR 0006 and ADR 0007 advanced `0.1` to `0.2` to `0.3` by
  regenerating in place. A directory fork here would preserve a shape that has never been emitted.
- **Add the interval as an optional field and leave the validators alone.** An optional interval on
  an evaluated verdict is exactly the reportable-but-unenforced shape this project keeps finding: a
  field that can be omitted will be omitted, and the gate that was supposed to require a sampling
  distribution would never fire.
- **Gate calibration on the point estimate and report the bound alongside.** The bound would then be
  recorded and never checked, which is a certificate filed and never presented.
