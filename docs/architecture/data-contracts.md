# Data contracts

All biological time is experiment-relative seconds. Wall-clock timestamps are provenance only.
Histories are canonicalized by `(time_seconds, event_id)` and reject duplicate IDs, mixed subjects,
and evidence after the request time.

Observations preserve raw/minimally processed values, assay metadata, quality, uncertainty, and an
explicit missingness state. Observed zero and missing are different. Censored and below-detection
records are different from technical failure. Below-detection records carry their detection limit;
one-sided censored records carry a direction and limit; interval-censored records carry ordered,
unit-consistent bounds. Their `value` remains empty so a bound is never mistaken for an observed or
imputed value.

Interventions separate intended exposure from actual perturbation realization. A guide assignment
or drug exposure is not interpreted as complete target engagement. Environment snapshots, lineage,
and neighborhood graphs are first-class inputs. Completeness declarations distinguish “no event”
from “no record supplied.”

Environment variable keys are case-folded to one canonical form. A backend must reject conflicting
assignments to the same variable and time unless its model defines an explicit precedence or fusion
rule; provenance IDs are never biological ordering rules. Realized-perturbation measurements cannot
predate the intervention whose efficacy they establish.

Lineage and population observations retain the target history subject plus an explicit evidence role
and source subject. Backends that cannot model sibling, ancestor, or population evidence must reject
it rather than treating it as a direct measurement.

Query outputs and forecast predictions declare units. Precision requirements bind a target and a
named horizon, and planning targets use typed quantities in those same units. A backend cannot make
numeric outputs comparable merely by assigning them the same ontology label.

Run `make schemas` after any contract change. A breaking semantic change requires a schema migration
and version decision rather than silently changing version `1.0`.

Provenance binds results to model/configuration and posterior-schema fingerprints, the query,
context, non-event history structure, and the exact content hash of every source event. Recursive
filtering rejects changed payloads hidden behind a previously assimilated event ID.

Sample-backed posteriors use an explicit two-axis convention: sample artifacts have shape
`(sample_count, state_dimension_count)` with axes `("sample", "state_dimension")`; optional weights
have one `"sample"` axis and exactly one entry per sample.

Public models prevent top-level field assignment, but Python mappings nested inside them are not
deeply frozen. Callers must treat nested mappings as read-only, and adapters must serialize and
revalidate contracts at process and storage boundaries before trusting their fingerprints.

## Public-real dataset manifests

`cellstate.data.DatasetManifest` is the **experimental** evidence boundary for biological data. Its
independent version is `0.1-experimental`; it does not change or claim stability under public domain
schema v1. A manifest requires a publicly downloadable real-data origin, remote acquired artifacts
with retrieval timestamps and SHA-256 hashes, source-scoped use terms, an experimental-unit
hierarchy, sampling/linkage semantics, modality alignment, and a scoped claim assessment.

Experimental units distinguish donor, organism, tissue, clone, culture, organoid, plate, well,
sample, cell, and spatial region. The manifest binds alignment keys and sampling subjects to those
units. It rejects a default split or biological-replicate unit finer than a shared sampling or
randomization unit, preventing cells from one well from silently becoming independent replicates.

Sampling explicitly distinguishes destructive endpoints, repeated population snapshots,
lineage-linked endpoints, partial nondestructive sampling, and longitudinal nondestructive
measurement. Destructive sampling cannot claim same-cell linkage. Supported and conditionally
supported claims carry a structured scope over subject, system boundary, biological system,
modalities, interventions, environments, functional outputs, cutoff, and horizons. Claim gates fail
closed when the scoped evidence, temporal coverage, alignment, controls, or replication are absent.

Use policies cover every source and explicitly assess research training, commercial training,
benchmark evaluation, source redistribution, derived-model distribution, and publication. A policy
record makes restrictions machine-readable; admission and later workflows must still choose an
intended use and reject `unknown` or `prohibited` status.

Dataset manifests describe source evidence; they do not download, transform, impute, pair, or
interpret measurements. Normalization, split, and training-run manifests will be separate,
content-addressed contracts. Do not create a dataset manifest with placeholder checksums, guessed
licensing, or aspirational eligibility.

The generated schema in `schemas/experimental/` captures structural JSON constraints. Several
scientific invariants require cross-field Pydantic validators and therefore **must** be revalidated
through the Python model; JSON-Schema-only acceptance is never an eligibility decision.
