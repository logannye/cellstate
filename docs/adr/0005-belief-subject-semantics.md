# ADR 0005: Typed belief subjects and schema v2

- **Status:** Accepted
- **Date:** 2026-08-09
- **Supersedes:** the generic-subject semantics of public schema v1

## Context

Schema v1 identifies histories and beliefs with a generic `subject_id`. The biological program
requires distinct semantics for an individually tracked cell, a clone or lineage, a sampled
population, and a spatial niche. Those are different estimands even when the source table contains
one row per cell.

Most single-cell assays are destructive. Cells sampled from a culture at different times are not
repeated observations of one living cell. Conversely, a viability-preserving assay such as
Live-seq may support an individual-cell belief and future outcome. A barcode may establish clone
membership without establishing exact parentage, and a spatial coordinate may establish a niche
relationship without establishing cell identity through time.

The experimental dataset manifest already distinguishes sampling design, but runtime schema v1
cannot carry that distinction into an estimation request, belief, forecast, or plan. It also cannot
represent evidence source and target separately, record collection effects, or bind output
aggregation to the belief subject. Automatically defaulting these fields would turn missing
scientific evidence into asserted linkage.

## Decision

The biological public boundary moves to **schema v2**. It uses a typed subject descriptor shared by
queries, histories, beliefs, forecasts, measurement decisions, and intervention plans. The existing
Python class names become the v2 API; v1 remains an immutable legacy serialization contract rather
than a second biological runtime.

### Belief-subject kinds

Schema v2 distinguishes at least:

1. **Individual cell** -- one cell whose identity is established by direct tracking or an explicitly
   viability-preserving sampling design.
2. **Clone or lineage** -- a set or branching process linked by an observed or probabilistic
   inheritance relation, barcode, phylogeny, or parent--child track.
3. **Population** -- a distribution of cells belonging to a declared experimental population,
   sample, well, culture, condition, or other aggregation unit.
4. **Spatial niche** -- a declared region plus its cell membership, neighborhood graph, and spatial
   context at a defined scale.

Every descriptor carries a stable subject identifier, kind, biological-system identity,
aggregation/membership semantics, and an identity or linkage basis. Confidence is explicit when
membership or lineage is inferred. Similar expression, shared cluster labels, nearest neighbors,
optimal transport, or adjacent sampling time never count as observed individual identity.

### Sampling subject is not belief subject

The schema keeps separate identifiers for:

- sampled biological entity;
- assay row or observation source;
- randomization and experimental unit;
- target belief subject;
- prediction target and aggregation;
- biological replicate and split unit.

A dataset containing destructively sampled individual cells may support a population-level belief.
It does not thereby support individual longitudinal dynamics. A population target may use cell-level
measurements while retaining well, replicate, and study hierarchy.

### Evidence links and collection effects

Each observation is attached to the target through a typed evidence link containing the source
subject, target subject, role, linkage basis, and confidence where applicable. Roles include direct,
ancestor, descendant, sibling, clone aggregate, matched population, general population, spatial
neighbor, and external reference. Backends advertise and enforce the roles they model.

Observation collection declares whether it is nondestructive, viability preserving with a known
effect, partially destructive sampling of a population, or terminal/destructive for the sampled
entity. A terminal observation of an individual cell closes that individual's direct history;
later direct observations require a different subject or an explicit proof that the collection was
not terminal. For population beliefs, destructive sampling changes the sampled membership and must
record its aggregation and sampling fraction when known.

Evidence duration and uncertain timing remain explicit. Sample collection, intervention and
environment start/stop, death, differentiation, migration, contact changes, and media changes are
first-class history events or typed event intervals rather than facts inferred from filenames.

### Output and downstream binding

Every query target, posterior factor, forecast, uncertainty statement, and decision result is bound
to the same typed belief subject and declares its aggregation. A backend cannot forecast an
individual trajectory from a population belief unless it exposes an explicit, supported change of
estimand. Subject projection or aggregation is a named model operation, not an implicit cast.

Forecast and planning support is assessed at the requested subject level. Causal and transport
status also binds source and target populations; it cannot be attached only to a model globally.

### Query-compiled state

Schema v2 includes a serialized compiled state specification. It identifies the active factors,
targets, subject, evidence roles, temporal resolution, bounded intervention/environment domains,
support requirements, and validity thresholds for one query. Beliefs contain exactly the required
active factor marginals plus any explicitly modeled nuisance, context, and perturbation-realization
blocks. Out-of-query dimensions are not mislabeled as unidentifiable.

## Version and migration policy

This is a breaking semantic change.

- Core v2 boundary objects use `schema_version = "2.0"` and are emitted under `schemas/v2/`.
- Checked-in `schemas/v1/` files remain immutable evidence of the former wire format.
- The package minor version advances when the v2 implementation lands.
- Public API functions accept v2 biological objects only. They never silently infer a subject kind,
  evidence link, collection effect, aggregation, support domain, or validity threshold.
- Dataset-manifest schema is independently versioned from the core runtime. ADR 0006 introduced
  `0.2-experimental` scoped claim/loss/metric assessments, and ADR 0007 advances the manifest to
  `0.3-experimental` for content-addressed slices and representability proofs. Manifest sampling
  units must still map explicitly into v2 runtime subjects.

Migration is deliberately fail-closed:

| v1 artifact | v2 treatment |
| --- | --- |
| Query | Requires caller-supplied subject, aggregation, bounded action/environment domains, and acceptance policy. |
| History/request | Requires source/target evidence links, collection effects, and event-interval annotations. |
| Belief | Must be re-estimated under v2; a posterior with unspecified subject and active-factor semantics is not relabeled. |
| Forecast | Must be regenerated from a v2 belief and scenario; causal/support status cannot be reconstructed from v1 numbers. |
| Plan | Must be regenerated under v2 readiness and candidate-support gates. |

The migration module may validate and inventory legacy v1 payloads and may construct v2 input
objects only from explicit migration annotations. It must reject incomplete annotations. There is no
automatic migration for produced beliefs, forecasts, or plans.

The synthetic reference backend migrates to v2 for contract tests. Its old serialized examples may
remain as legacy fixtures, but they do not become biological v2 evidence.

## Consequences

- Public-real adapters remain blocked from emitting biological beliefs until they can construct
  typed subjects and evidence links without guessing.
- Existing v1 client payloads require deliberate migration; this is acceptable during pre-alpha and
  safer than preserving scientifically ambiguous defaults.
- Query compilation, support/readiness checks, perturbation realization, causal status, and the
  fourth measurement-decision operation are part of the coordinated v2 boundary implementation.
- Subject-specific backends may share interfaces and components without claiming that their states
  or evidence are interchangeable.
- The Vertical A query stays unfrozen until a destructive K562 study maps to a population subject
  and Live-seq maps to an individual subject through reviewed representability proofs.

## Acceptance criteria

The v2 implementation is complete only when tests prove that it:

- rejects a destructive cross-sectional time course presented as one individual history;
- permits destructive sampled cells to inform a typed population belief through an explicit link;
- permits genuinely viability-preserving longitudinal evidence for one individual cell;
- records uncertain clone or lineage membership without asserting parentage;
- rejects unsupported evidence roles and implicit subject casts;
- binds target units, aggregation, forecasts, uncertainty, and planning to the subject;
- preserves v1 schemas without silently accepting them as v2; and
- refuses to relabel a v1 belief, forecast, or plan as a v2 scientific artifact.
