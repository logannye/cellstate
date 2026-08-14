# Belief state

The object this project exists to compute. A belief is not evidence of anything by itself: it becomes
*faithful* only by passing the two tests in [predictive sufficiency](predictive-sufficiency.md), and
no belief has yet been emitted by a biological model.

## The estimand

Cells are partially observed, stochastic, history-dependent dynamical systems, so the answer to "what
state is this cell system in?" is a distribution, not a point. For a query `Q` at experimental time
`t`:

```text
B_t^Q = P(X_t^Q, Theta, R_{<=t}, Xi | H_{<=t}, C, Q)
```

- `X_t^Q` — the query-relevant dynamic state;
- `Theta` — stable or slowly varying subject, donor, genotype, and lineage parameters;
- `R_{<=t}` — realized perturbation or target engagement, distinct from intended assignment;
- `Xi` — assay, batch, capture, segmentation, and other measurement nuisance;
- `H_{<=t}` — the causally ordered evidence history;
- `C` — declared population, lineage, spatial, and environmental context.

Forecasting is a separate operation over the same object,
`P(Z_{t+h} | B_t^Q, do(U_{t:t+h}), E_{t:t+h}, Q)`, where `Z` holds declared query targets and never
backend-private latent coordinates. The derivation is in
[`architecture/full-buildout.md`](../architecture/full-buildout.md#formal-estimand).

## The belief is compiled from the query

There is no universal state and no minimal state outside an exact query. The state required to
predict survival after a ten-minute signaling perturbation is not the state required to predict
differentiation over two weeks.

`CompiledStateSpecification` (`src/cellstate/domain/specification.py`) records which factor families
are active, each with a rationale, and which are excluded, each with a rationale; the two sets must
partition the eight families of `StateFactor`: stable identity, slow memory, regulation, signaling,
metabolism, physical structure, damage and stress, and functional capacity. Nothing is silently
absent.

That specification is fingerprinted against the query and travels inside every belief and forecast,
so a consumer cannot read a number without reading the support it was computed under.

Factor marginals do not make the factors independent. The joint posterior is authoritative, and a
belief is rejected unless each parametric factor, context, realization, and nuisance posterior equals
the corresponding marginal of that joint.

## Subjects

Most single-cell assays destroy the cell they measure, so a time course of different cells is a
sequence of population distributions, not an observed trajectory. `SubjectKind`
(`src/cellstate/domain/subjects.py`) therefore types every belief, and the observed identity evidence
determines which type may be claimed:

| Subject | Claimable when |
| --- | --- |
| individual cell | the same cell was measured more than once nondestructively |
| clone or lineage | a heritable barcode, phylogeny, or observed parentage links the members |
| population | membership is an experimental sample, condition, well, or library |
| spatial niche | coordinates, contacts, or a neighborhood graph were measured |

Subject identity may never be inferred from expression similarity, nearest neighbors, optimal
transport, shared cluster membership, or adjacent sampling times.

A population belief is a complete instance of the object, not a compromise: it is hidden, is inferred
from evidence, evolves under intervention, and is subject to both faithfulness tests. The first
faithful representation this project targets is a population belief.

## What a belief must carry

`CellStateBelief` (`src/cellstate/domain/belief.py`) is not a vector with metadata attached. It is
rejected at construction unless it carries all of:

- a typed subject compatible with the query's subject specification, and an `as_of` time;
- the query itself, with agreeing query, history, and context fingerprints;
- the compiled state specification, checked against the query's targets, horizons, admissible
  evidence roles, and acceptance thresholds;
- a joint posterior whose dimensions equal the compiled state — as distribution parameters, as
  weighted samples or particles held in content-addressed artifacts, or as an explicit
  `UnavailableDistribution`;
- one factor belief per active compiled factor, each with its timescales, its evidence status
  (`observed`, `inferred_with_support`, `unidentifiable`), and evidence event IDs that appear in
  provenance;
- a posterior over realized intervention effect covering every compiled realization dimension, kept
  distinct from intended assignment;
- a posterior over `Xi` nuisance whenever the compiled state declares nuisance dimensions;
- an uncertainty breakdown containing measurement, biological, parameter, model, and counterfactual
  components, exactly once each;
- diagnostics — support, sufficiency, identifiability, decision uncertainty, calibration, and causal
  support — each separating whether it was evaluated from whether it passed, each bound to the
  query's own thresholds, with identifiability classifying every active joint dimension;
- a readiness report whose validity flags are re-derived and fail closed, with explicit reasons
  whenever abstention is required;
- provenance: model identity and fingerprint, posterior schema, seed, and one SHA-256 fingerprint per
  source event and per cited validation artifact.

## What a belief may not do

- **A posterior mean is a summary, not the state.** Callers propagate the belief, not its mean.
- **Missing is not zero.** An unavailable posterior must be marked unidentifiable, and an evaluated
  scalar refuses a numeric value unless it is supported. Absence is typed, never imputed.
- **Nuisance is not promoted to biology.** `Xi` is inferred and kept separate, not corrected away.
- **Structural completeness is not scientific validity.** `BeliefStatus` reports only whether the
  posterior blocks are present. Whether the belief may be used is the readiness report's answer;
  whether it is faithful is a question only the two tests answer.
- **`do(U)` in the notation is not a causal guarantee.** Every output labels its evidential status:
  predictive association, identified population effect, transported effect under enumerated
  assumptions, mechanistic extrapolation, or unsupported.

## Status

No biological backend is registered and no belief has been emitted by a biological model. The
linear-Gaussian reference produces contract-shaped beliefs to exercise these rules in tests; its
numbers are examples of contract behavior, not estimates of cellular state.
