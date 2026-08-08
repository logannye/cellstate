# cellstate

`cellstate` estimates a **query-conditioned probability distribution** over hidden, causally
relevant cellular state. It does not define state as a cell label, cluster, UMAP coordinate,
transcriptomic embedding, pathway score, or deterministic latent vector.

The target abstraction is a belief state:

\[
B_t = P(X_t, \Theta \mid Y_{\le t}, U_{<t}, E_{\le t}, L_t, C)
\]

Its adequacy is empirical: once the belief is known, raw history should add little information
about future functional behavior under the interventions, environments, and horizons named by the
query.

> **Research scaffold:** the repository contains contracts and a linear-Gaussian reference model
> for software integration tests. It contains no biologically or clinically validated cell model.

## What is included

- Frozen-top-level, strict, JSON-schema-versioned contracts for queries, events, context, lineage,
  neighborhood graphs, beliefs, forecasts, and plans.
- A canonical event history that preserves timing, provenance, explicit missingness, intended
  interventions, and measured/inferred perturbation efficacy.
- Separate ports for state estimation, controlled evolution, and intervention choice.
- Query-target predictive distributions on forecasts, alongside the propagated latent posterior.
- Structured state factors across identity, memory, regulation, signaling, metabolism, physical
  structure, damage/stress, and functional capacity.
- Joint distributions rather than point-only outputs; explicit measurement, biological,
  parameter, model, and counterfactual uncertainty.
- Explicit `observed`, `inferred_with_support`, `unidentifiable`, `unsupported`, and
  `not_evaluated` states instead of numeric placeholders.
- A Kalman-style reference backend with controlled dynamics, recursive updates, posterior
  propagation, reproducible sampling, and explicit abstention where scientific support is absent.
- Scientific validation primitives, including the state-only vs. state-plus-history sufficiency
  diagnostic.

## Quick start

```bash
uv sync --all-extras --no-editable
uv run --no-editable python examples/estimate_state.py
uv run --no-editable pytest
```

The public API requires an explicit model; there is deliberately no scientifically meaningless
default:

```python
from cellstate import estimate_cell_state
from cellstate.reference import LinearGaussianReference, minimal_reference_config

model = LinearGaussianReference(minimal_reference_config())
belief = estimate_cell_state(request, estimator=model)
```

Then propagate the **full posterior**, not only its mean:

```python
from cellstate import evolve_cell_state

forecast = evolve_cell_state(
    belief,
    scenario=scenario,
    evolution_model=model,
)

for prediction in forecast.target_predictions:
    print(prediction.target.term.label, prediction.distribution)
```

See [`examples/estimate_state.py`](examples/estimate_state.py) for a complete executable request.

## Architecture

```text
StateQuery + canonical CellHistory
                │
                ▼
       CellStateEstimator port
                │
                ▼
 CellStateBelief (joint posterior)
       │                    │
       ▼                    ▼
StateEvolutionModel   InterventionPlanner
       │                    │
       ▼                    ▼
 StateForecast        InterventionPlan
```

The stable domain package is backend-neutral. PyTorch, JAX, AnnData, Zarr, experiment tracking,
and artifact stores should enter through adapters rather than leak into boundary schemas.

Future scenarios must make active-intervention and environment persistence explicit: inherit the
belief's current exposure/context, clear it intentionally, or provide replacement events at scenario
start. This prevents missing future controls from silently becoming zero exposure.

Each forecast covers one named query horizon. Intervention objectives name that same horizon and
declared query targets; candidates with different start times, durations, or horizons cannot be
compared. Planners score typed target predictions rather than internal latent coordinates.

Pydantic freezes assignment to public models, but nested mapping values are not deeply immutable.
Treat them as read-only, and serialize/revalidate contracts when they cross process or storage
boundaries. Provenance carries a model/configuration fingerprint and posterior-schema identifier so
recursive updates cannot silently mix incompatible latent semantics. It also carries exact hashes
for assimilated event payloads and the non-event history structure, preventing an event ID from
being reused with changed evidence during recursive filtering.

### Reference-backend boundary

The included linear-Gaussian backend is intentionally narrow and fails closed. It requires complete
past intervention, environment, lineage, and neighborhood/contact records, because an unknown
history cannot honestly be treated as “no event.” It supports only the declared synthetic species
guard, modalities, units, continuous controls, and global environment effects. It rejects lineage,
population, contact, division, regional environment,
target-specific controls, censored likelihoods, and assay metadata or uncertainty forms that its
likelihood does not model. These contracts remain available for real backends; the reference
implementation does not pretend to condition on them. Its prior is anchored to an explicit
`prior_time_seconds` in the model configuration, so adding a missing-assay record cannot move the
prior epoch or change the posterior.

## Scientific non-goals

- No universal/minimal cellular state claim.
- No implicit missing-as-zero imputation.
- No automatic batch “removal” or forced reference-class assignment.
- No causal claim from perturbation labels alone.
- No claim that the current transcriptome is Markovian.
- No silent support for unimplemented factors, hazards, or environments.
- No validation based only on cluster appearance or random held-out cells from familiar studies.

Read [the architecture](docs/architecture/overview.md), [the inference
pipeline](docs/architecture/inference-pipeline.md), and [the scientific validation
contract](docs/validation/scientific-validation.md) before adding a biological backend.

## Repository policy

Large assay tensors, genomic data, donor-identifiable data, and model weights stay outside Git and
are represented by content-addressed `ArtifactRef` objects. Publishing, package ownership, a remote,
and an open-source license are intentionally not configured; those are owner decisions.
