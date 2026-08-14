# cellstate

`cellstate` computes a query-conditioned probability distribution over hidden, causally relevant
cellular state, and forecasts future molecular and functional behavior under declared interventions
and environments. The purpose of the project is a faithful and accurate representation of cell
state; everything else exists to make that representation trustworthy.

No single embedding, assay, or dataset is treated as universal cell state. The end-state is a family
of support-bounded biological backends trained and evaluated against traceable public real-cell
experiments.

Start with:

1. the [belief-state concept](concepts/belief-state.md);
2. the [predictive-sufficiency concept](concepts/predictive-sufficiency.md);
3. the [full-build architecture](architecture/full-buildout.md);
4. the [scientific validation contract](validation/scientific-validation.md); and
5. the [roadmap](roadmap.md), which is the sole authority for implementation order, the
   state-capability ledger, and graduation status.

## Status

Phase 1 is active: make the faithfulness tests executable and measure the observational floor.

Phase 0 is complete. Schema v2 enforces typed subjects, destructive evidence, bounded query support,
query-compiled state, perturbation realization, scientific readiness and abstention, causal status,
and standalone decision-oriented measurement selection. Manifest `0.3-experimental` carries
claim-specific eligibility, content-addressed slices, and machine-checked representability ledgers,
which validate bound declarations and attestations without resolving source bytes or replaying
selectors.

The `gse274113` RNA observation model **is** registered and beliefs **have** been emitted from real
human CD34+ cells; see [the exploration guide](guides/explore-the-system.md). **No benchmark is
scientifically admitted and the eligibility ledger is 0 of 10**, and on this substrate it will stay
there: the deposit's CRISPRi arm is a measured null, so the capability tests divide by a
perturbation signal that was never created.

Every `metric_id` the frozen sci-Plex3 suite declares now resolves to an executable implementation,
and both faithfulness tests execute and return a verdict with a grouped bootstrap interval. What has
not happened is their application to biology: the frozen artifact's own metric bindings remain
`specification_only`, nothing outside `tests/` invokes either harness, no baseline has been scored
against any other, and the observational floor is unmeasured. **The repository can now recognize a
faithful representation on supplied arrays; it has still produced no scientific number about a
cell.**

The sci-Plex3 K562 24-hour component remains a non-executable `SCAFFOLD` and its benchmark remains
`COMPONENT_BENCHMARK`, not admitted. Its estimand has no admissible pre-cutoff observation and one
horizon, so it cannot carry a sufficiency verdict; it is retained as data and split infrastructure
and as the Phase 1 metric proving ground. Its held-out endpoint values, outcomes, and scoring
authority remain sealed.

The protected-execution control plane is suspended and the rank-16 candidate family is retired for
the state path; see [ADR 0013](adr/0013-state-first-roadmap-reordering.md).
