# ADR 0013: Reorder the roadmap around cell state as the primary object

- **Status:** Accepted
- **Date:** 2026-08-12

## Context

The purpose of this project is to compute a faithful and accurate representation of hidden cellular
state. Everything else exists to make that representation trustworthy.

The roadmap this record supersedes did not order work that way. Three properties of the superseded
plan are decisive.

First, the only real-data vertical it scheduled cannot carry the property that defines the purpose.
[ADR 0009](0009-population-response-component-boundary.md) established — correctly and unforced —
that the sci-Plex3 K562 24-hour component is an endpoint regression and "not a posterior over hidden
state at the inference cutoff," and that hidden-state estimation "remains unreachable from component
admission." Its frozen query declares no admissible pre-cutoff evidence and one horizon. With an
empty pre-cutoff history the predictive-sufficiency comparison is not merely unrun; it is
inapplicable, because `M2` and `M1` receive the same inputs. The bundle contract records this
directly: the `sufficiency_evaluator` port is `unsupported`, with the rationale that "one
context-to-endpoint experiment cannot establish hidden-state sufficiency." The persistence and
temporal state-space baselines — the two comparisons that would demonstrate a state carries
information — are declared inapplicable for the same reason.

Second, the two tests that define faithfulness have no executable implementation. The metric suite is
specification-only, the sufficiency and calibration functions have no caller outside tests, and no
baseline has ever been scored against another. The project cannot currently recognize a faithful
representation, or an unfaithful one.

Third, the plan's sequencing authority was not respected in practice. The numbered queue ended at
item 12; subsequent execution-containment and authorization work was authorized by unnumbered
paragraphs appended after it, at least one of which was written by the same commit that implemented
what it authorized. That work — container containment, reproducible runtime distribution, and a
protected-execution control plane — advances no capability that a cell-state representation requires.

## Decision

Reorder the roadmap so that cell state is the primary object and every scheduled item is instrumental
to it.

1. **Adopt a state-capability ledger (S1–S10)** as the sole measure of progress, each entry worded so
   a reader can objectively determine whether it is satisfied. A query for which S1 (admissible
   pre-cutoff evidence) or S3 (at least two horizons) is structurally unavailable cannot test S7 (an
   executable sufficiency verdict) and is not scheduled as a step toward the purpose.
2. **Adopt a purpose test.** Every queue item names the ledger capabilities it advances. An item that
   advances none is not scheduled.
3. **Require authorization to precede implementation.** A commit that amends the roadmap may not also
   implement the work it authorizes.
4. **Restrict status-conferring content to a single ordered queue.** Appended prose confers no
   authorization. Queue IDs are prefixed `Q` and do not continue the historical Item 1–12 sequence,
   whose numbering is bound into content-addressed manifests and must not be reused.
5. **Require an ADR for any change to phase order, the ledger, or a graduation gate.** This record
   satisfies that rule for this reordering.
6. **Establish that a negative verdict graduates a phase.** A phase whose gate is a measurement passes
   by producing the measurement with its interval. The gate reads `evaluation_status` and the presence
   of an interval, never `outcome`.

### Phase mapping

| Superseded | Current | Change |
| --- | --- | --- |
| Phase 0 — scope and benchmark semantics | Phase 0 | Unchanged; complete |
| Phase 1 — reproducible public-real-data foundation | Phase 2 deliverables | Bronze/silver/gold storage, ontologies, adapters, golden slices, and both reproducibility gates are retained and folded into the phase that admits the state-bearing source |
| — | **Phase 1 — make the faithfulness tests executable and measure the floor** | New, and now first. The project cannot recognize its own success until the tests exist |
| — | **Phase 2 — freeze a state-bearing estimand** | New. Selecting evidence that can carry the tests is scheduled ahead of modeling |
| Phase 2 — observation models and posterior infrastructure | Phase 3 | Retargeted onto the state-bearing estimand's paired modalities |
| Phase 3 — first population-response backend | Phase 4 — first state backend and the sufficiency verdict | The gate now includes an executed sufficiency verdict and risk-coverage monotonicity |
| Phase 4 — temporal dynamics, events, branching | Phase 5 | Unchanged in substance |
| Phase 5 — multimodal, environmental, spatial | Phase 6 | Unchanged in substance |
| Phase 6 — translational T-cell vertical | Phase 7 | Unchanged in substance |
| Phase 7 — measurement value and planning | Phase 8 | Unchanged in substance |

### Reclassifications

- **The sci-Plex3 K562 24-hour component is reclassified off the state path.** It is a completed
  engineering exercise that proved the data and split machinery, and it is the proving ground for the
  Phase 1 metric implementations. Its partitions, membership arrays, manifests, split discipline,
  definition-time leakage review, golden fixtures, and six fitted `p1` baseline states remain in
  scope. Its candidate lifecycle does not. Nothing about this reclassification asserts a defect in
  that work; ADR 0008 and ADR 0009 stand as written.
- **The protected-execution authorization control plane is suspended.** It governs execution of a
  candidate for a component that cannot emit a cell-state belief, so it advances no ledger capability.
  No proposal is to be approved and no protected execution dispatched. The suspension is enforced by a
  fail-closed first step in the dispatch workflow, not by prose alone.
- **The rank-16 continuous admixture candidate family is retired for the state path.** It is a
  condition-level mean model whose latent is indexed at the well, with a free parameter per observed
  action that cannot generalize to an unseen action in principle. Its corrected objective mathematics,
  its equal-unit normalization constant, and its effective-context diagnostic are carried into the
  Phase 4 model guidance; the lifecycle scaffolding is not.

## Consequences

- The next actions are `Q1` and `Q2` — implement the metric suite and the sufficiency and calibration
  harnesses — neither of which requires new data, new authorization, or a successful model fit.
- `Q3`, measuring the observational floor on the proving-ground query, is blocked. Scoring the fitted
  `p1` baselines against one another requires protected-partition predictions, and `p2`/`p3`/`p4` are
  sealed behind the lifecycle grants of [ADR 0011](0011-sciplex3-p1-loader-and-baselines.md), issued
  only by the control plane this record suspends. Unblocking it requires a further ADR that either
  authorizes a single held-out read for baseline-versus-baseline scoring with no candidate model in
  the loop, or moves the floor measurement onto the Phase 2 estimand.
- Adding a bootstrap interval to `SufficiencyReport` and a coverage-error upper bound to the
  calibration report are serialized-contract changes. Each needs a schema-version decision,
  regenerated JSON Schemas, and round-trip tests.
- The Phase 2 gate on `sufficiency_evaluator` is not satisfiable today: the port is never derived into
  `required_ports`, and a hand-typed `required` disposition outside the derived set is rejected.
  Deriving it is a prerequisite of that gate and is named as such.
- The metric suite frozen by [ADR 0008](0008-sciplex3-k562-component-benchmark.md) is not
  retrofitted. Its five metric families stay frozen; the differential-expression-weighted and
  rank-based metric requirements bind suites frozen from this record forward.
- Program rules 1 and 3 are enforced by `tests/test_roadmap_queue_contract.py`. Rules 2 and 4 are not
  yet mechanically enforced; the enforcing mechanism is a diff-aware pull-request job, and until it
  exists those rules are conventions.
- This record and the roadmap change it authorizes contain no `src/` implementation, so rule 3 of the
  superseded ordering and rule 2 of the current one are both satisfied: rule 2 binds every change from
  here forward.

## Rejected alternatives

- **Continue to the protected execution and decide afterwards.** The run is one-use and nonissuing;
  at maximum success it emits a bounded terminal report and destroys the model with the runner. It
  would produce no evidence about cell state either way.
- **Retrofit the frozen sci-Plex3 benchmark with a pre-cutoff observation.** No pre-treatment
  molecular observation of the same population exists in that source. Relabeling 24-hour vehicle wells
  as a time-zero reference is the exact confusion ADR 0009 rejects.
- **Keep the superseded phase order and reprioritize informally.** Informal reprioritization is what
  produced the unnumbered-paragraph authorizations. The ordering must be the written one.
- **Declare the sufficiency test satisfied on a query where the history is empty.** Under an empty
  history the gain is identically zero and the test passes trivially. A gate that cannot fail is not
  a gate.
